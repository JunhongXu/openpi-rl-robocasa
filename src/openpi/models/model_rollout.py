import dataclasses
import jax.numpy as jnp
import pathlib

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi_client import image_tools

from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
import numpy as np
import gymnasium as gym
import imageio
import collections


config = _config.get_config("pi0_robocasa_finetune_target_atomic_seen")
checkpoint_dir = pathlib.Path(download.maybe_download(
    "/media/Data/models/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"
))
print("action dim", config.model.action_dim)
# Resolve norm stats from the checkpoint's own assets dir instead of the training soup.
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
print("action dim", config.model.action_dim)
print("config", config.model)
print("config.data", config.data)
policy = policy_config.create_trained_policy(config, checkpoint_dir)
img = jnp.ones((224, 224, 3), dtype=jnp.uint8)
wrist = jnp.ones((224, 224, 3), dtype=jnp.uint8)
state = jnp.ones((10,))
prompt = "task"
obs = {
    "observation/image": img,
    "observation/wrist_image": wrist,
    "observation/state": state,
    "prompt": prompt,
}
output = policy.infer(obs, stochastic=True)
trajectory = output['trajectory']
vt_sampled = trajectory['vt_sampled']
times = trajectory['times']
old_logprob = trajectory['logpdf']
print("times", times.shape)
print("vt_sampled", vt_sampled.shape)
print("old_logprob", old_logprob.shape)
logprob = policy._compute_action_logprob(
    output["observation"], trajectory["x_t"], vt_sampled, times
)["logprob"]
print("logprob", logprob.shape)
print(jnp.allclose(logprob, old_logprob))

print("logprob", logprob)
print("old_logprob", old_logprob)


def rollout_robocasa(policy):
    env = gym.make("robocasa/OpenCabinet", split='pretrain', seed=0)
    obs, info = env.reset()
    task_lang = obs["annotation.human.task_description"]
    replay_images = []
    action_plan = collections.deque()
    replan_steps = 5
    for t in range(500):
        if not action_plan:
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, 224, 224)
                )
            state = np.concatenate(
                (
                    obs["state.end_effector_position_relative"],
                    obs["state.end_effector_rotation_relative"],
                    obs["state.base_position"],
                    obs["state.base_rotation"],
                    obs["state.gripper_qpos"],
                ), axis=0
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": task_lang,
            }
            output = policy.infer(element, stochastic=True)
            action_chunk = output["actions"]
            assert (
                len(action_chunk) >= replan_steps
            ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
            action_plan.extend(action_chunk[: replan_steps])
        action = convert_action(action_plan.popleft())
        obs, reward, done, truncated, info = env.step(action)
        done = info["success"] # for robocasa, usuccess entry in info
        replay_img = env.render()
        replay_img = np.ascontiguousarray(replay_img)
        replay_img = image_tools.convert_to_uint8(
            replay_img
        )
        # if t % 2 == 0 or t == horizon - 1 or done:
        replay_images.append(replay_img)
        if done:
            # task_successes += 1
            # total_successes += 1
            print("Done")
            break
    imageio.mimwrite(
        pathlib.Path("./rollouts") / "rollout.mp4",
        [np.asarray(x) for x in replay_images],
        fps=20,
    )

rollout_robocasa(policy)
