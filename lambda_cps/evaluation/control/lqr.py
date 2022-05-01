import numpy as np
from control import lqr

from lambda_cps.envs.wrapped import WrapperBase


class LQRController:
    def __init__(self, K: np.ndarray):
        self.K = K

    def predict(self, x: np.ndarray) -> np.ndarray:
        return -self.K @ x


def build_lqr_controller(env: WrapperBase,
                         stable_state: np.ndarray,
                         Q: np.ndarray,
                         R: np.ndarray,
                         traj_len: int = 10,
                         n_traj: int = 500,
                         step_size: float = 0.05,
                         lr: float = 1e-3,
                         n_opt_iter: int = 1000
                         ) -> LQRController:
    A, B = env.linearize_at(stable_state, traj_len, n_traj, step_size, lr, n_opt_iter)
    K, _, _ = lqr(A, B, Q, R)

    return LQRController(K)
