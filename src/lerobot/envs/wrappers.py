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
        x = float(rng.integers(100, 412))
        y = float(rng.integers(100, 412))
        theta = float(rng.uniform(0, 2 * np.pi))
        self.env.unwrapped.goal_pose = np.array([x, y, theta], dtype=np.float64)
        info["goal_pose"] = self.env.unwrapped.goal_pose
        return obs, info
