"""Environment rollout utils for pi0 flow-matching policy on RoboCasa.
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
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action  # pyright: ignore[reportMissingImports]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_policy_and_model(checkpoint_dir: pathlib.Path):
    config = _config.get_config("pi0_robocasa_finetune_target_atomic_seen")
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
# Rollout
# ---------------------------------------------------------------------------


def rollout_one_episode(
    env,
    policy,
    max_num_steps: int,
    replan_steps: int,
    save_video_path: pathlib.Path | None = None,
):
    """Run one episode, push one transition per inference (chunk) into `buffer`.

    Sparse reward: 0 every chunk except the one in which `info["success"]` flips
    True, where it's set to 1.0.
    """
    env_obs, _ = env.reset()
    task_lang = env_obs["annotation.human.task_description"]

    record_video = save_video_path is not None
    if save_video_path is not None:
        replay_images: list[np.ndarray] = []
    chunks_used = 0
    success = False

    action_plan = collections.deque()

    buffer = {
        "observation": [],
        "action": [],
        "reward": [],
        "done": [],
        "truncated": [],
        "success": [],
        "logprob": [],
        "entropy": [],
        "vt_mean": [],
        "vt_sampled": [],
        "x_t": []
    }
    # profile the time taken for each step
    import time
    for _ in range(max_num_steps):
        raw = env_obs_to_inputs(env_obs, task_lang)
        if not action_plan:
            # Computing the new action chunk
            policy_output = policy.infer(raw, stochastic=True)
            observation = policy_output["observation"]
            flow_traj = policy_output["trajectory"]
            action_chunk = policy_output["actions"]
            action_plan.extend(action_chunk[: replan_steps])
            # We only extract the information needed for policy update at each replanning step.
            buffer["observation"].append(observation)
            buffer["action"].append(action_chunk)
            buffer["reward"].append(0.0)
            buffer["done"].append(False)
            buffer["truncated"].append(False)
            buffer["success"].append(False)
            # Policy information
            buffer["logprob"].append(flow_traj["logprob"])
            buffer["entropy"].append(flow_traj["entropy"])
            buffer["vt_mean"].append(flow_traj["vt_mean"])
            buffer["vt_sampled"].append(flow_traj["vt_sampled"])
            buffer["x_t"].append(flow_traj["x_t"])

        action = action_plan.popleft()
        action = convert_action(action)
        env_obs, _, done, truncated, info = env.step(action)
        success = info.get("success", False)
        if record_video:
            replay_img = env.render()
            replay_img = np.ascontiguousarray(replay_img)
            replay_img = image_tools.convert_to_uint8(
                replay_img
            )
            replay_images.append(replay_img)

        if truncated or done:
            break
    if record_video:
        save_video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(save_video_path), [np.asarray(x) for x in replay_images], fps=20)

    return {"success": success, "buffer": buffer}


def rollout_episodes(
    env, policy, buffer, horizon, replan_steps, num_episodes, task_name, save_video_path
):
    for i in range(num_episodes):
        result = rollout_one_episode(
            env,
            policy,
            horizon,
            replan_steps,
            save_video_path=pathlib.Path(f"{save_video_path}/{task_name}/episode_{i}.mp4"),
        )
        buffer.add(result["buffer"])


if __name__ == "__main__":
    policy = load_policy_and_model(
        pathlib.Path(
            "/media/Data/models/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"
        )
    )
    env = gym.make("robocasa/OpenCabinet", split="pretrain", seed=0)
    task_horizon = get_task_horizon("OpenCabinet")
    # set dataset path and horizon
    horizon = int(task_horizon * 1.5) # the policy moves slow so give the policy extra time
    import time
    start_time = time.time()
    replay_buffer = ReplayBuffer(capacity=10000, num_envs=1, sample_transition=buffer)
    for i in range(10):
        result = rollout_one_episode(
            env,
            policy,
            horizon,
            5,
            save_video_path=pathlib.Path(f"./rollouts/open_cabinet_{i}.mp4"),
        )
        print(f"Rollout {i} success {result['success']}")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
