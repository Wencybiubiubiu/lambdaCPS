from typing import List

import numpy as np

from lambda_cps.envs.wrapped import WrapperBase


class RandomShootingController:
    def __init__(self, env: WrapperBase):
        self.env = env

    def next_actions(self, n_step: int, n_traj: int) -> List[np.ndarray]:
        current_state = self.env.get_state()

        best_reward = -np.inf
        best_action_seq = None
        for i in range(n_traj):
            self.env.set_state(current_state)
            rollout_reward = 0
            action_seq = []
            for _ in range(n_step):
                action = self.env.action_space.sample()
                _, reward, _, _ = self.env.step(action)
                action_seq.append(action)
                rollout_reward += reward
            if rollout_reward > best_reward:
                best_reward = rollout_reward
                best_action_seq = action_seq

        self.env.set_state(current_state)

        return best_action_seq
