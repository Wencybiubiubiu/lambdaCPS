import copy

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


if __name__ == '__main__':
    test_random_shooting()
