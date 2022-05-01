from abc import abstractmethod
from typing import Tuple

import gym
import numpy as np
import torch as th

from lambda_cps.config import DEFAULT_DEVICE
from lambda_cps.envs.gym.pendulum import PendulumEnv


class WrapperBase(gym.Wrapper):
    @abstractmethod
    def set_state(self, state: np.ndarray):
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_param(self, param_name: str, param_value: float):
        pass

    @abstractmethod
    def get_param(self, param_name: str) -> float:
        pass

    def linearize_at(self,
                     state: np.ndarray,
                     traj_len: int = 10,
                     n_traj: int = 100,
                     step_size: float = 0.05,
                     lr: float = 1e-3,
                     n_opt_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample-based linearizing. This function is heavy and not suitable running online.
        :param state: state linearize around
        :param traj_len: trajectory length
        :param n_traj: total sampled trajectory number
        :param step_size: step size of discrete time system
        :param lr: learning rate for regression
        :param n_opt_iter: optimization step for regression
        :return: A, B matrix
        """
        current_state = self.get_state()

        x_0 = []
        a = []
        x_1 = []

        for _ in range(n_traj):
            for _ in range(traj_len):
                x_0.append(self.get_state())
                action = self.action_space.sample()
                a.append(action)
                self.set_state(state)
                self.step(action)
                x_1.append(self.get_state())

        state_matrix = th.zeros([self.observation_space.shape[0]] * 2,
                                dtype=th.float32,
                                device=DEFAULT_DEVICE,
                                requires_grad=True)
        action_matrix = th.zeros([self.observation_space.shape[0], self.action_space.shape[0]],
                                 dtype=th.float32,
                                 device=DEFAULT_DEVICE,
                                 requires_grad=True)

        x_0 = th.tensor(np.array(x_0), dtype=th.float32, device=DEFAULT_DEVICE)
        a = th.tensor(np.array(a), dtype=th.float32, device=DEFAULT_DEVICE)
        x_1 = th.tensor(np.array(x_1), dtype=th.float32, device=DEFAULT_DEVICE)

        opt = th.optim.Adam(params=[state_matrix, action_matrix], lr=lr)
        loss_fn = th.nn.MSELoss()

        for _ in range(n_opt_iter):
            opt.zero_grad()
            loss = loss_fn(x_0 + step_size * (x_0 @ state_matrix.T + a @ action_matrix.T), x_1)
            loss.backward()
            opt.step()

        self.set_state(current_state)

        return state_matrix.detach().cpu().numpy(), action_matrix.detach().cpu().numpy()


class Pendulum(WrapperBase):

    def __init__(self):
        super(Pendulum, self).__init__(PendulumEnv())

    def set_state(self, state: np.ndarray):
        self.env.state = state

    def get_state(self) -> np.ndarray:
        return self.env.state

    def set_param(self, param_name: str, param_value: float):
        setattr(self.env, param_name, param_value)

    def get_param(self, param_name: str) -> float:
        return getattr(self.env, param_name)
