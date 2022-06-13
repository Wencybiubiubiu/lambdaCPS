import numpy as np
from lambda_cps.envs import Pendulum
from lambda_cps.evaluation.control.lqr import build_lqr_controller


class Controller:

    def __init__(self):
        return

    # input: None
    # output: environment and controller
    # Controller can take state as input and output the action
    # Environment can take action as a step and output the next state
    # action = controller.predict(obs)
    # obs, r, _, _ = env.step(action)
    # obs is the vector to represent the state
    def get_env_and_controller(self):
        env = Pendulum()
        env.reset()
        Q = 100 * np.eye(2)
        R = 0.1 * np.eye(1)
        stable_state = np.array([0, 0])

        lqr_controller = build_lqr_controller(env, stable_state, Q, R)

        return env, lqr_controller

    # update environment with the new generated design
    # input: environment, design
    # output: new environment
    def set_new_design_to_env(self, input_env, input_design, input_xml):
        new_env = input_env

        # Input design will have the format of [adjacency_matrix, feature_matrix]
        # Here, a pendulum is [ [[0, 1], [1, 0]], [[0, 0], [mass, length]]]
        # mass and length is the parameter of the single hinge component
        # xml file is None now.
        # If it needs a better format, we can adjust it
        new_env.set_param('m', input_design[1][1][0])
        new_env.set_param('l', input_design[1][1][1])
        return new_env
