import numpy as np

from lambda_cps.envs import Pendulum


def test_linearize_at():
    env = Pendulum()
    env.reset()

    a, b = env.linearize_at(np.array([0.0, 0.0]), step_size=1, n_traj=500, traj_len=20, lr=1e-3, n_opt_iter=2000)
    print(a)
    print(b)


if __name__ == '__main__':
    test_linearize_at()
