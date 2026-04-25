"""PPO training loop for the pi0 flow-matching policy on RoboCasa.

Sparse reward = success bit at episode end. No value head yet — we use Monte Carlo
returns as the advantage (compute_gae with value=0 and gae_lambda=1.0 reduces to
that). One PPO transition = one inference call (one observation + K flow steps);
the env executes `replan_steps` actions from each chunk before we replan.

Why not use `policy.infer` for both rollout and training?
    `Policy._sample_actions` is jitted with the model's state frozen at construction
    time (see `nnx_utils.module_jit`). After we update params with the optimizer,
    that wrapper would still use the old params. So we bypass it and call the model
    directly via `nnx.jit`, which re-traces against the live `model` graph.
"""

import collections
import dataclasses
import logging
import pathlib

import flax.nnx as nnx
import gymnasium as gym
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi_client import image_tools

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.rl.ppo_loss import ppo_loss
from openpi.rl.replay_buffer import ReplayBuffer
from openpi.shared import download
from openpi.training import config as _config

from robocasa.utils.env_utils import convert_action

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclasses.dataclass
class PPOConfig:
    iterations: int = 100

    # Rollout
    num_envs: int = 1
    max_chunks_per_episode: int = 50
    replan_steps: int = 5
    num_flow_steps: int = 10
    env_name: str = "robocasa/OpenCabinet"
    env_split: str = "pretrain"

    # PPO update
    learning_rate: float = 1e-5
    ppo_epochs: int = 4
    minibatch_size: int = 16
    clip_ratio: float = 0.2
    entropy_coef: float = 0.0
    log_ratio_clip: float = 20.0
    gamma: float = 0.99

    # Misc
    seed: int = 0
    rollout_video_every: int = 5
    video_dir: pathlib.Path = pathlib.Path("./rollouts")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_policy_and_model():
    config = _config.get_config("pi0_robocasa_finetune_target_atomic_seen")
    checkpoint_dir = pathlib.Path(download.maybe_download(
        "/media/Data/models/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"
    ))
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            assets=_config.AssetsConfig(
                assets_dir=str(checkpoint_dir / "assets"),
                asset_id=config.data.assets.asset_id,
            ),
        ),
    )
    return policy_config.create_trained_policy(config, checkpoint_dir)


# ---------------------------------------------------------------------------
# Env <-> policy plumbing
# ---------------------------------------------------------------------------

def env_obs_to_inputs(env_obs, task_lang):
    img = np.ascontiguousarray(env_obs["video.robot0_agentview_left"])
    wrist_img = np.ascontiguousarray(env_obs["video.robot0_eye_in_hand"])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
    state = np.concatenate(
        (
            env_obs["state.end_effector_position_relative"],
            env_obs["state.end_effector_rotation_relative"],
            env_obs["state.base_position"],
            env_obs["state.base_rotation"],
            env_obs["state.gripper_qpos"],
        ),
        axis=0,
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": task_lang,
    }


def transform_obs(raw_obs, input_transform):
    """Apply the policy's input pipeline; return Observation pytree with batch=1."""
    inputs = jax.tree.map(lambda x: x, raw_obs)
    inputs = input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    return _model.Observation.from_dict(inputs)


# ---------------------------------------------------------------------------
# State-aware jitted entry points (re-trace against current model params)
# ---------------------------------------------------------------------------

@nnx.jit(static_argnames=("num_steps", "stochastic"))
def jit_sample_actions(model, rng, observation, *, num_steps, stochastic):
    return model.sample_actions(rng, observation, num_steps=num_steps, stochastic=stochastic)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_one_episode(env, model, input_transform, output_transform, buffer,
                        cfg: PPOConfig, rng, save_video_path: pathlib.Path | None = None):
    """Run one episode, push one transition per inference (chunk) into `buffer`.

    Sparse reward: 0 every chunk except the one in which `info["success"]` flips
    True, where it's set to 1.0.
    """
    env_obs, _ = env.reset()
    task_lang = env_obs["annotation.human.task_description"]

    replay_images = [] if save_video_path is not None else None
    chunks_used = 0
    success = False

    for _ in range(cfg.max_chunks_per_episode):
        raw = env_obs_to_inputs(env_obs, task_lang)
        observation = transform_obs(raw, input_transform)

        rng, sample_rng = jax.random.split(rng)
        sampled = jit_sample_actions(
            model, sample_rng, observation,
            num_steps=cfg.num_flow_steps, stochastic=True,
        )

        # Apply output transforms (unnormalize, etc.) on the unbatched dict — same
        # contract as policy.infer.
        actions_for_env = output_transform({
            "state": np.asarray(observation.state[0]),
            "actions": np.asarray(sampled["x_0"][0]),
        })["actions"]

        traj = sampled["trajectory"]
        # After the swapaxes inside sample_actions:
        #   x_t, vt_sampled : [B=1, K, ah, ad]
        #   times           : [K]
        #   logpdf          : [B=1, K]
        x_t = np.asarray(traj["x_t"])
        vt_sampled = np.asarray(traj["vt_sampled"])
        times_K = np.asarray(traj["times"])
        old_logprob = np.asarray(traj["logpdf"])

        chunk_reward = 0.0
        chunk_done = False
        chunk_truncated = False
        for step_idx in range(min(cfg.replan_steps, len(actions_for_env))):
            action = convert_action(actions_for_env[step_idx])
            env_obs, _, done, truncated, info = env.step(action)
            if save_video_path is not None:
                replay_images.append(np.ascontiguousarray(env.render()))
            if info.get("success", False):
                success = True
                chunk_reward = 1.0  # sparse success reward
                chunk_done = True
                break
            if done:
                chunk_done = True
                break
            if truncated:
                chunk_truncated = True
                break

        # Buffer leaves must have leading axis = num_envs = 1.
        obs_leaves = jax.tree.map(np.asarray, observation)
        transition = {
            "observation": obs_leaves,
            "x_t": x_t,
            "vt_sampled": vt_sampled,
            "times": np.broadcast_to(times_K[None, :], (1, cfg.num_flow_steps)).copy(),
            "old_logprob": old_logprob,
            "reward": np.asarray([chunk_reward], dtype=np.float32),
            # No value head: zeros + compute_gae(lambda=1.0) -> Monte Carlo returns.
            "value": np.zeros((1,), dtype=np.float32),
            "done": np.asarray([float(chunk_done)], dtype=np.float32),
            "truncated": np.asarray([float(chunk_truncated)], dtype=np.float32),
        }
        buffer.add(transition)
        chunks_used += 1

        if chunk_done or chunk_truncated:
            break

    if save_video_path is not None and replay_images:
        save_video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(save_video_path, replay_images, fps=20)

    return {"length_chunks": chunks_used, "success": success, "rng": rng}


def build_buffer_from_sample(model, input_transform, env, cfg: PPOConfig):
    """One throwaway inference to derive every leaf shape, then allocate the buffer."""
    env_obs, _ = env.reset()
    task_lang = env_obs["annotation.human.task_description"]
    raw = env_obs_to_inputs(env_obs, task_lang)
    observation = transform_obs(raw, input_transform)

    sampled = jit_sample_actions(
        model, jax.random.key(0), observation,
        num_steps=cfg.num_flow_steps, stochastic=True,
    )
    traj = sampled["trajectory"]

    sample_transition = {
        "observation": jax.tree.map(np.asarray, observation),
        "x_t": np.asarray(traj["x_t"]),
        "vt_sampled": np.asarray(traj["vt_sampled"]),
        "times": np.broadcast_to(np.asarray(traj["times"])[None, :], (1, cfg.num_flow_steps)).copy(),
        "old_logprob": np.asarray(traj["logpdf"]),
        "reward": np.zeros((1,), dtype=np.float32),
        "value": np.zeros((1,), dtype=np.float32),
        "done": np.zeros((1,), dtype=np.float32),
        "truncated": np.zeros((1,), dtype=np.float32),
    }
    return ReplayBuffer(
        capacity=cfg.max_chunks_per_episode,
        num_envs=cfg.num_envs,
        sample_transition=sample_transition,
    )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def make_update_step(cfg: PPOConfig):
    @nnx.jit
    def update_step(model, optimizer, batch):
        def loss_fn(model):
            obs = batch["observation"]  # Observation pytree, leaves [batch_size, ...]
            result = model.compute_action_logprob(
                obs, batch["x_t"], batch["vt_sampled"], batch["times"]
            )
            loss, info = ppo_loss(
                new_logprob=result["logprob"],
                old_logprob=batch["old_logprob"],
                advantage=batch["advantage"],
                entropy=result["entropy"],
                clip_ratio=cfg.clip_ratio,
                log_ratio_clip=cfg.log_ratio_clip,
                entropy_coef=cfg.entropy_coef,
            )
            return loss, info

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(grads)
        return loss, info

    return update_step


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(cfg: PPOConfig | None = None):
    cfg = cfg or PPOConfig()
    np_rng = np.random.default_rng(cfg.seed)
    rng = jax.random.key(cfg.seed)

    log.info("Loading model and policy...")
    policy = load_policy_and_model()
    model = policy.model
    input_transform = policy._input_transform
    output_transform = policy._output_transform

    log.info("Setting up env...")
    env = gym.make(cfg.env_name, split=cfg.env_split, seed=cfg.seed)

    log.info("Building optimizer + buffer...")
    optimizer = nnx.Optimizer(model, optax.adamw(cfg.learning_rate))
    buffer = build_buffer_from_sample(model, input_transform, env, cfg)
    update_step = make_update_step(cfg)

    success_history: collections.deque[float] = collections.deque(maxlen=20)

    for it in range(cfg.iterations):
        # ---- Rollout ----
        buffer.reset()
        video_path = (
            cfg.video_dir / f"iter_{it:04d}.mp4"
            if cfg.rollout_video_every and it % cfg.rollout_video_every == 0
            else None
        )
        stats = rollout_one_episode(
            env, model, input_transform, output_transform, buffer, cfg, rng,
            save_video_path=video_path,
        )
        rng = stats["rng"]
        success_history.append(float(stats["success"]))

        # ---- Returns (Monte Carlo via compute_gae with value=0, lambda=1) ----
        last_value = np.zeros((cfg.num_envs,), dtype=np.float32)
        buffer.compute_gae(last_value, gamma=cfg.gamma, gae_lambda=1.0)

        # ---- PPO updates ----
        losses, kls, clip_fracs = [], [], []
        for batch in buffer.iterate_minibatches(np_rng, cfg.minibatch_size, cfg.ppo_epochs):
            loss, info = update_step(model, optimizer, batch)
            losses.append(float(loss))
            kls.append(float(info["approx_kl"]))
            clip_fracs.append(float(info["clip_frac"]))

        log.info(
            "iter=%4d  chunks=%2d  success=%d  recent_succ=%.2f  "
            "loss=%+.4f  kl=%.4f  clip=%.3f",
            it, stats["length_chunks"], int(stats["success"]),
            float(np.mean(success_history)) if success_history else 0.0,
            float(np.mean(losses)) if losses else 0.0,
            float(np.mean(kls)) if kls else 0.0,
            float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        )


if __name__ == "__main__":
    main()
