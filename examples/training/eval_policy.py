# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrates how to evaluate a trained Diffusion Policy on the PushT environment."""

from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch


class RandomGoalWrapper(gym.Wrapper):
    """Randomizes the PushT goal pose (position + angle) on each reset.

    The default env always uses goal_pose = [256, 256, pi/4]. This wrapper
    samples a random goal from the seeded RNG after each reset, enabling
    evaluation on a broader distribution of goals.
    """

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        rng = self.env.unwrapped.np_random
        x = float(rng.integers(206, 306))
        y = float(rng.integers(206, 306))
        theta = float(rng.uniform(-np.pi, np.pi))
        self.env.unwrapped.goal_pose = np.array([x, y, theta], dtype=np.float64)
        info["goal_pose"] = self.env.unwrapped.goal_pose
        return obs, info


from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    # Path to the saved policy checkpoint (output of train_policy.py).
    pretrained_path = Path(
        "outputs/train/diffusion_pusht_augmented/checkpoints/440000/pretrained_model"
    )

    # Select your device.
    device = torch.device("cuda")

    # Number of episodes to evaluate.
    n_episodes = 20
    # Image resolution must match what the policy was trained on.
    # Check pretrained_path/policy_preprocessor.json -> normalizer_processor -> features -> observation.image -> shape
    observation_height = 96
    observation_width = 96

    # Load policy weights and config from checkpoint.
    policy = DiffusionPolicy.from_pretrained(pretrained_path)
    policy.eval()
    policy.to(device)

    # Load preprocessor and postprocessor from checkpoint.
    # This restores the normalization stats used during training.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
        },
    )

    # Directory to save evaluation videos.
    video_dir = Path("outputs/eval/diffusion_pusht_v4/videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create the PushT gym environment.
    # observation_height/width must match training resolution to avoid feature map mismatch.
    # visualization_height/width controls the rendered video resolution (can be larger).
    import gym_pusht  # noqa: F401 - registers gym_pusht environments

    env = RandomGoalWrapper(
        gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=384,
            visualization_height=384,
        )
    )

    successes = []
    sum_rewards = []
    max_rewards = []

    for ep in range(n_episodes):
        # Reset the policy's internal observation/action queues at the start of each episode.
        policy.reset()

        # Use seeds from the training distribution (dataset used seeds 0-200).
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        ep_max_reward = 0.0
        success = False
        frames = []

        while not done:
            # Capture rendered frame for video (uses visualization_width/height).
            frames.append(env.render())

            # Convert gym observation (numpy) to lerobot tensor format.
            # preprocess_observation handles channel conversion (HWC->CHW), uint8->float32, batch dim.
            obs_dict = preprocess_observation(obs)

            # Normalize observations using training stats and move to device.
            obs_dict = preprocessor(obs_dict)

            # Run policy inference (no gradients needed).
            with torch.inference_mode():
                action = policy.select_action(obs_dict)

            # Denormalize action back to environment scale.
            action = postprocessor(action)

            # Convert to numpy and remove batch dimension for single env.
            action_np = action.to("cpu").numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]  # (action_dim,)

            # Step the environment.
            obs, reward, terminated, truncated, info = env.step(action_np)

            ep_reward += reward
            ep_max_reward = max(ep_max_reward, reward)
            success = info.get("is_success", False)
            done = terminated or truncated

        # Save episode video.
        video_path = (
            video_dir / f"episode_{ep:02d}_{'success' if success else 'fail'}.mp4"
        )
        imageio.mimsave(str(video_path), frames, fps=10)

        successes.append(success)
        sum_rewards.append(ep_reward)
        max_rewards.append(ep_max_reward)
        print(
            f"Episode {ep + 1:2d}: sum_reward={ep_reward:7.2f}  max_reward={ep_max_reward:.4f}  success={success} "
        )

    env.close()

    print(f"\n=== Evaluation Results ({n_episodes} episodes) ===")
    print(f"Success rate:    {sum(successes) / n_episodes * 100:.1f}%")
    print(f"Avg sum reward:  {sum(sum_rewards) / n_episodes:.2f}")
    print(f"Avg max reward:  {sum(max_rewards) / n_episodes:.4f}")


if __name__ == "__main__":
    main()
