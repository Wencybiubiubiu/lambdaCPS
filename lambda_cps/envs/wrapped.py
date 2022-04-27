from abc import abstractmethod

import gym
import numpy as np
from lambda_cps.envs.gym.pendulum import PendulumEnv


class WrapperBase(gym.Wrapper):
    @abstractmethod
    def set_state(self, state: np.ndarray):
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass


class Pendulum(WrapperBase):
    def __init__(self):
        super(Pendulum, self).__init__(PendulumEnv())

    def set_state(self, state: np.ndarray):
        self.env.state = state

    def get_state(self) -> np.ndarray:
        return self.env.state
