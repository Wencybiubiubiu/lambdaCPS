import copy

import numpy as np

from lambda_cps.evaluation.control.lqr import build_lqr_controller
from lambda_cps.evaluation.control.random_shooting import RandomShootingController
from lambda_cps.envs import Pendulum

def test_random_shooting():
    env = Pendulum()
    env_model = copy.deepcopy(env)
    rs_controller = RandomShootingController(env_model)

    env.reset()
    rew_sum = 0
    for i in range(200):
        env_model.set_state(env.get_state())
        action = rs_controller.next_actions(20, 100)[0]
        _, r, _, _ = env.step(action)
        rew_sum += r
        env.render()

    print(rew_sum)


def test_lqr():
    env = Pendulum()
    env.reset()
    #print(env.get_state())

    Q = 100 * np.eye(2)
    R = 0.1 * np.eye(1)
    stable_state = np.array([0, 0])

    lqr_controller = build_lqr_controller(env, stable_state, Q, R)
    print(lqr_controller.K)

    obs = env.reset()
    rew_sum = 0
    for i in range(200):

        action = lqr_controller.predict(obs)

        obs, r, _, _ = env.step(action)
        rew_sum += r
        env.render()

    print(rew_sum)


if __name__ == '__main__':
    # test_random_shooting()
    test_lqr()
