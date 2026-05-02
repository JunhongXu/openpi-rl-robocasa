"""PPO-style RL trainer for the pi0 flow-matching policy on RoboCasa.

Sharding mirrors ``scripts/train.py`` (see ``.claude/sharding_for_rl.md``):
  * 2D mesh (batch, fsdp).
  * ``TrainState`` built once under ``fsdp_sharding``.
  * Two jitted entrypoints share ``state_sharding``: the actor forward (rollouts)
    and the RL update — never re-shard params between them.
  * The outer loop runs inside ``sharding.set_mesh(mesh)`` so the activation
    sharding constraints inside ``gemma.py`` / ``siglip.py`` actually fire.

Known gaps (left as TODOs rather than silently papered over):
  * ``rollout_one_episode`` stores ``vt_sampled``/``x_t`` but not the per-step
    flow times. They're regenerated deterministically here from ``num_steps``;
    if ``sample_actions`` ever changes its time schedule, fix it in both places.
  * No critic. ``_compute_advantages`` falls back to a Monte-Carlo return; once a
    value head is added, swap to ``self.replay_buffer.compute_gae(...)``.
"""

import dataclasses
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from openpi.policies import policy_config
from openpi.rl.ppo_loss import ppo_loss
from openpi.rl.replay_buffer import ReplayBuffer
from openpi.rl.rollout_utils import rollout_one_episode
from robocasa.utils.dataset_registry_utils import get_task_horizon
from openpi.rl.rollout_utils import obs_to_dict
from openpi.rl.rollout_utils import dict_to_obs 
from flax import traverse_util
import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class RLTrainerConfig:
    # Pretrained checkpoint to fine-tune from.
    train_config_name: str
    checkpoint_path: pathlib.Path

    # Tasks.
    training_tasks: list[str]
    eval_tasks: list[str] = dataclasses.field(default_factory=list)

    # Mesh — fsdp_devices must divide jax.device_count().
    fsdp_devices: int = 1

    # Schedule.
    num_train_steps: int = 1000
    seed: int = 0

    # Rollout / buffer.
    num_envs: int = 1
    rollouts_per_step: int = 1
    replan_steps: int = 5
    horizon_multiplier: float = 1.5
    num_flow_steps: int = 10
    replay_buffer_capacity: int = 4096

    # PPO.
    minibatch_size: int = 32
    num_epochs: int = 4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True

    # Eval / IO.
    eval_every_n_steps: int = 50
    eval_episodes_per_task: int = 5
    video_dir: pathlib.Path | None = None

    # Specifies which weights should be frozen.
    freeze_filter: nnx.filterlib.Filter = dataclasses.field(default_factory=nnx.Nothing)

class RLTrainer:
    """PPO trainer with FSDP-sharded params for the pi0 flow-matching policy."""

    def __init__(self, config: RLTrainerConfig):
        self.config = config
        self.train_config = _config.get_config(config.train_config_name)

        if config.minibatch_size % jax.device_count() != 0:
            raise ValueError(
                f"minibatch_size {config.minibatch_size} must be divisible by "
                f"jax.device_count()={jax.device_count()}."
            )

        self.mesh = sharding.make_mesh(config.fsdp_devices)
        self.data_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        self.replicated_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec()
        )

        self.policy = policy_config.create_trained_policy(
            self.train_config, config.checkpoint_path
        )
        self.train_state, self.train_state_sharding = self._init_train_state()
        # Re-bind the rollout policy's jit wrappers to the freshly sharded params.
        self._sync_policy_params()

        self._p_rl_update = jax.jit(
            self._rl_update,
            in_shardings=(
                self.replicated_sharding,
                self.train_state_sharding,
                self.data_sharding,
            ),
            out_shardings=(self.train_state_sharding, self.replicated_sharding),
            donate_argnums=(1,),
        )

        self.replay_buffer: ReplayBuffer | None = None
        self.np_rng = np.random.default_rng(config.seed)
        self.train_rng = jax.random.key(config.seed)
        self._envs: dict[str, tuple[Any, int]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_train_state(
        self,
    ) -> tuple[training_utils.TrainState, training_utils.TrainState]:
        tx = _optimizer.create_optimizer(
            self.train_config.optimizer,
            self.train_config.lr_schedule,
            weight_decay_mask=None,
        )
        graphdef = nnx.graphdef(self.policy.model)

        def _make_state() -> training_utils.TrainState:
            params = nnx.state(self.policy.model)
            all_params_flat = traverse_util.flatten_dict(params.to_pure_dict(), sep="/")                                                                                                            
            print(
                f"{len(all_params_flat)} all params leaves, {sum(int(np.prod(v.shape)) for v in all_params_flat.values())/1e6:.2f}M params"
            )
            for k in list(all_params_flat):                                                                                                                                                       
                print(" ", k, all_params_flat[k].shape)   
            trainable = nnx.state(self.policy.model).filter(nnx.Not(self.config.freeze_filter))                                                                                       
            flat = traverse_util.flatten_dict(trainable.to_pure_dict(), sep="/")                                                                                                            
            print(
                f"{len(flat)} trainable leaves, {sum(int(np.prod(v.shape)) for v in flat.values())/1e6:.2f}M params"
            )
            for k in list(flat):                                                                                                                                                       
                print(" ", k, flat[k].shape)   
            return training_utils.TrainState(
                step=jnp.array(0, dtype=jnp.int32),
                params=params,
                model_def=graphdef,
                tx=tx,
                opt_state=tx.init(params.filter(nnx.Not(self.config.freeze_filter))),
                ema_decay=None,
                ema_params=None,
            )

        train_state_shape = jax.eval_shape(_make_state)
        state_sharding = sharding.fsdp_sharding(train_state_shape, self.mesh, log=True)
        train_state = jax.jit(_make_state, out_shardings=state_sharding)()
        jax.block_until_ready(train_state)
        return train_state, state_sharding

    def _sync_policy_params(self) -> None:
        """Copy current ``train_state.params`` into the rollout policy and rebuild
        its jit wrappers so subsequent ``policy.infer()`` calls see new weights.

        ``nnx_utils.module_jit`` snapshots state at construction time, so after
        every PPO update we have to rewire the policy's sample/logprob wrappers.
        """
        nnx.update(self.policy.model, self.train_state.params)
        self.policy._sample_actions = nnx_utils.module_jit(
            self.policy.model.sample_actions, static_argnames=("stochastic",)
        )
        self.policy._compute_action_logprob = nnx_utils.module_jit(
            self.policy.model.compute_action_logprob
        )

    # ------------------------------------------------------------------
    # Environments / rollouts
    # ------------------------------------------------------------------

    def _get_env(self, task_name: str):
        if task_name not in self._envs:
            env = gym.make(
                f"robocasa/{task_name}", split="pretrain", seed=self.config.seed
            )
            horizon = int(get_task_horizon(task_name) * self.config.horizon_multiplier)
            self._envs[task_name] = (env, horizon)
        return self._envs[task_name]

    def collect_data(self, step: int) -> list[dict]:
        summaries = []
        for ep in range(self.config.rollouts_per_step):
            task_name = self.config.training_tasks[
                (step * self.config.rollouts_per_step + ep)
                % len(self.config.training_tasks)
            ]
            env, horizon = self._get_env(task_name)
            video_path = self._video_path("train", task_name, step, ep)
            result = rollout_one_episode(
                env, self.policy, horizon, self.config.replan_steps,
                save_video_path=video_path,
            )
            self._push_episode_to_buffer(result)
            summaries.append({"task": task_name, "success": float(result["success"])})
        return summaries

    def _push_episode_to_buffer(self, result: dict) -> None:
        """Convert a rollout into per-timestep transitions with shape
        ``[num_envs, ...]`` and add them to the replay buffer."""
        buffer = result["buffer"]
        T = len(buffer["action"])
        if T == 0:
            return

        # Sparse reward: episode-success → +1.0 on the final stored chunk.
        rewards = np.zeros((T,), dtype=np.float32)
        dones = np.zeros((T,), dtype=np.float32)
        truncs = np.zeros((T,), dtype=np.float32)
        if bool(result["success"]):
            rewards[-1] = 1.0
            dones[-1] = 1.0
        else:
            truncs[-1] = 1.0

        # Per-step flow times — must match `sample_actions`'s schedule.
        K = self.config.num_flow_steps
        dt = -1.0 / K
        times = (1.0 + dt * np.arange(K, dtype=np.float32))[None, :]  # [1, K]

        sample = self._build_sample_transition(buffer, times)
        if self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(
                capacity=self.config.replay_buffer_capacity,
                num_envs=self.config.num_envs,
                sample_transition=sample,
            )

        for t in range(T):
            self.replay_buffer.add(
                self._index_transition(buffer, rewards, dones, truncs, times, t)
            )

    def _build_sample_transition(self, buffer: dict, times: np.ndarray) -> dict:
        """Produce a single ``[num_envs, ...]`` transition used to allocate the
        replay buffer storage."""
        return self._index_transition(
            buffer,
            rewards=np.zeros((1,), dtype=np.float32),
            dones=np.zeros((1,), dtype=np.float32),
            truncs=np.zeros((1,), dtype=np.float32),
            times=times,
            t=0,
        )

    def _index_transition(
        self,
        buffer: dict,
        rewards: np.ndarray,
        dones: np.ndarray,
        truncs: np.ndarray,
        times: np.ndarray,
        t: int,
    ) -> dict:
        # `observation` is a concatenated Observation pytree; slice along axis 0.
        obs_t = jax.tree.map(lambda x: x[t : t + 1], buffer["observation"])
        # The remaining leaves were appended per-chunk with leading dim [1, ...].
        return {
            "observation": obs_t,
            "action": np.asarray(buffer["action"][t])[None, ...],
            "x_t": np.asarray(buffer["x_t"][t]),                # [1, K, ah, ad]
            "vt_sampled": np.asarray(buffer["vt_sampled"][t]),  # [1, K, ah, ad]
            "vt_mean": np.asarray(buffer["vt_mean"][t]),        # [1, K, ah, ad]
            "logprob": np.asarray(buffer["logprob"][t]),        # [1, K]
            "entropy": np.asarray(buffer["entropy"][t]),        # [1, K]
            "times": times,                                      # [1, K]
            "reward": rewards[t : t + 1],
            "done": dones[t : t + 1],
            "truncated": truncs[t : t + 1],
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _rl_update(
        self,
        rng: jax.Array,
        state: training_utils.TrainState,
        batch: dict,
    ) -> tuple[training_utils.TrainState, dict]:
        del rng  # PPO update is deterministic given the batch.
        model = nnx.merge(state.model_def, state.params)
        model.eval()

        def loss_fn(model, batch):
            out = model.compute_action_logprob(
                batch["observation"],
                batch["x_t"],
                batch["vt_sampled"],
                batch["times"],
            )
            loss, info = ppo_loss(
                new_logprob=out["logprob"],
                old_logprob=batch["logprob"],
                advantage=batch["advantage"],
                entropy=out["entropy"],
                clip_ratio=self.config.clip_ratio,
                entropy_coef=self.config.entropy_coef,
                normalize_advantage=self.config.normalize_advantage,
            )
            return loss, info

        diff_state = nnx.DiffState(0, self.train_config.trainable_filter)
        (loss, info), grads = nnx.value_and_grad(
            loss_fn, argnums=diff_state, has_aux=True
        )(model, batch)

        params = state.params.filter(self.train_config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        nnx.update(model, new_params)
        new_full_params = nnx.state(model)

        info = {**info, "loss": loss, "grad_norm": optax.global_norm(grads)}
        new_state = dataclasses.replace(
            state,
            step=state.step + 1,
            params=new_full_params,
            opt_state=new_opt_state,
        )
        return new_state, info

    def update_step(self) -> dict:
        """Compute advantages, then run ``num_epochs`` of minibatch updates."""
        assert self.replay_buffer is not None and self.replay_buffer.size > 0, (
            "update_step called with empty buffer"
        )
        self._compute_advantages()
        last_info: dict[str, Any] = {}
        for batch in self.replay_buffer.iterate_minibatches(
            self.np_rng,
            batch_size=self.config.minibatch_size,
            num_epochs=self.config.num_epochs,
        ):
            batch = jax.device_put(batch, self.data_sharding)
            self.train_rng, sub = jax.random.split(self.train_rng)
            self.train_state, last_info = self._p_rl_update(
                sub, self.train_state, batch
            )
        self._sync_policy_params()
        self.replay_buffer.reset()
        return last_info

    def _compute_advantages(self) -> None:
        """Critic-free Monte-Carlo return. Replace with
        ``self.replay_buffer.compute_gae(last_value=...)`` once a value head
        exists in the schema."""
        T = self.replay_buffer.size
        N = self.config.num_envs
        rewards = self.replay_buffer.data["reward"][:T].astype(np.float32)
        dones = self.replay_buffer.data["done"][:T].astype(np.float32)
        truncs = self.replay_buffer.data["truncated"][:T].astype(np.float32)
        returns = np.zeros((T, N), dtype=np.float32)
        running = np.zeros((N,), dtype=np.float32)
        for t in reversed(range(T)):
            # Reset the carry on either a real terminal or a time-limit cut.
            running = (1.0 - np.maximum(dones[t], truncs[t])) * running
            running = rewards[t] + self.config.gamma * running
            returns[t] = running
        if "advantage" not in self.replay_buffer.data:
            self.replay_buffer.data["advantage"] = np.zeros(
                (self.replay_buffer.capacity, N), dtype=np.float32
            )
            self.replay_buffer.data["returns"] = np.zeros(
                (self.replay_buffer.capacity, N), dtype=np.float32
            )
        self.replay_buffer.data["advantage"][:T] = returns
        self.replay_buffer.data["returns"][:T] = returns

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def train(self) -> None:
        for step in range(self.config.num_train_steps):
            with sharding.set_mesh(self.mesh):
                summaries = self.collect_data(step)
                info = self.update_step()
            log.info(
                "step %d | success=%.2f | loss=%.4f | kl=%.4f | clip=%.3f",
                step,
                float(np.mean([s["success"] for s in summaries])),
                float(info.get("loss", 0.0)),
                float(info.get("approx_kl", 0.0)),
                float(info.get("clip_frac", 0.0)),
            )
            if step > 0 and step % self.config.eval_every_n_steps == 0:
                self.evaluate(step)

    def evaluate(self, step: int) -> dict[str, float]:
        results: dict[str, float] = {}
        for task in self.config.eval_tasks:
            env, horizon = self._get_env(task)
            successes = 0
            for ep in range(self.config.eval_episodes_per_task):
                video_path = self._video_path("eval", task, step, ep)
                out = rollout_one_episode(
                    env, self.policy, horizon, self.config.replan_steps,
                    save_video_path=video_path,
                )
                successes += int(out["success"])
            results[task] = successes / max(1, self.config.eval_episodes_per_task)
        log.info("eval @ step %d: %s", step, results)
        return results

    def _video_path(
        self, kind: str, task: str, step: int, ep: int
    ) -> pathlib.Path | None:
        if self.config.video_dir is None:
            return None
        return self.config.video_dir / kind / task / f"step_{step:06d}_ep{ep}.mp4"
