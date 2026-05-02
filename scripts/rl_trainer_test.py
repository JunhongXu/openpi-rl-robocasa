"""Fast smoke test for ``openpi.training.rl_trainer.RLTrainer``.

Runs the smallest possible PPO loop end-to-end against a real RoboCasa env:
  * 1 train step
  * 1 rollout per step on a single short-horizon task
  * 1 minibatch / 1 epoch update
  * Tiny replay buffer
  * No eval, no video

Usage:
    uv run python scripts/rl_trainer_test.py \
        --checkpoint /media/Data/models/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000

The first compile of the actor + update will dominate wall time; this script's
purpose is to exercise the wiring (sharding, train state init, rollout buffer
schema, ppo update jit, param sync), not to actually train.
"""

import argparse
import logging
import pathlib
import time

from openpi.training.rl_trainer import RLTrainer
from openpi.training.rl_trainer import RLTrainerConfig
import flax.nnx as nnx
import openpi.shared.nnx_utils as nnx_utils


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path(
            "/media/Data/models/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"
        ),
        help="Pretrained pi0 checkpoint dir.",
    )
    p.add_argument(
        "--train_config_name",
        type=str,
        default="pi0_robocasa_finetune_target_atomic_seen",
    )
    p.add_argument("--task", type=str, default="OpenCabinet")
    p.add_argument("--fsdp_devices", type=int, default=1)
    p.add_argument("--num_flow_steps", type=int, default=10)
    p.add_argument("--horizon_multiplier", type=float, default=0.25,
                   help="Cap each episode hard so rollouts stay short.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    freeze_filter = nnx.All(                                                                                                                                                     
        nnx.Param,                                            
        nnx.Any(nnx_utils.PathRegex("PaliGemma.*"))
    )     
    config = RLTrainerConfig(
        train_config_name=args.train_config_name,
        checkpoint_path=args.checkpoint,
        training_tasks=[args.task],
        eval_tasks=[],
        fsdp_devices=args.fsdp_devices,
        num_train_steps=1,
        seed=0,
        # Rollout — keep it tiny.
        num_envs=1,
        rollouts_per_step=1,
        replan_steps=5,
        horizon_multiplier=args.horizon_multiplier,
        num_flow_steps=args.num_flow_steps,
        replay_buffer_capacity=64,
        # PPO — minimum work.
        minibatch_size=1,
        num_epochs=1,
        clip_ratio=0.2,
        entropy_coef=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=False,
        # No eval, no video.
        eval_every_n_steps=10**9,
        eval_episodes_per_task=0,
        video_dir=None,
        freeze_filter=freeze_filter 
    )

    t0 = time.monotonic()
    trainer = RLTrainer(config)
    logging.info("Trainer init took %.1fs", time.monotonic() - t0)

    # --- 1) collect ---
    t0 = time.monotonic()
    summaries = trainer.collect_data(step=0)
    logging.info(
        "collect_data: %.1fs | summaries=%s | buffer.size=%d",
        time.monotonic() - t0,
        summaries,
        0 if trainer.replay_buffer is None else trainer.replay_buffer.size,
    )
    assert trainer.replay_buffer is not None, "replay buffer was not allocated"
    assert trainer.replay_buffer.size > 0, "no transitions were stored"

    # Schema sanity — every key the PPO update reads must be present.
    required = {
        "observation", "x_t", "vt_sampled", "times",
        "logprob", "reward", "done", "truncated",
    }
    missing = required - set(trainer.replay_buffer.data.keys())
    assert not missing, f"replay buffer missing keys: {missing}"

    # --- 2) update ---
    from openpi.training import sharding
    starting_step = int(trainer.train_state.step)
    t0 = time.monotonic()
    with sharding.set_mesh(trainer.mesh):
        info = trainer.update_step()
    logging.info(
        "update_step: %.1fs | step=%d | info=%s",
        time.monotonic() - t0,
        int(trainer.train_state.step),
        {k: float(v) for k, v in info.items()},
    )
    assert int(trainer.train_state.step) > starting_step, (
        "train_state.step did not advance"
    )
    assert "loss" in info and "approx_kl" in info, (
        f"ppo info missing expected keys: {info.keys()}"
    )

    # --- 3) end-to-end driver, one extra step ---
    config_extra = trainer.config
    config_extra.num_train_steps = 1
    t0 = time.monotonic()
    trainer.train()
    logging.info("trainer.train(1 step): %.1fs", time.monotonic() - t0)

    logging.info("OK — RLTrainer smoke test passed.")


if __name__ == "__main__":
    main()
