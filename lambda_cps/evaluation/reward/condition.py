# handles pre/post-condition
# initial_state_generator (pre-condition) = initial_state
# (in pendulum example, if pre-condition is theta = [60,90], generator should random(theta))
# post_condition_check (post-condition, result_state) = | alpha - result_state (angle) |

import numpy as np
from gym.utils import seeding


class ConditionProcessor:

    def __init__(self, pre_condition, post_condition):
        self.pre_condition = pre_condition
        self.post_condition = post_condition
        self.np_random, seed = seeding.np_random(None)

        self.LOW = 0
        self.HIGH = 1

    def initial_state_generator(self):
        new_state = self.np_random.uniform(low=self.pre_condition[self.LOW], high=self.pre_condition[self.HIGH])

        return new_state

    # It will return absolute value of difference between result_state to its nearest range
    # This is a deprecated old version.
    def old_check_post_condition(self, result_state):
        check_low = np.abs(self.post_condition[self.LOW] - result_state)
        check_high = np.abs(self.post_condition[self.HIGH] - result_state)

        max_diff = np.array([check_low, check_high]).min(axis=0)

        return max_diff

    # It will return absolute value of difference between result_state to its nearest range
    def score_calculator(self, result_state):

        # set absolute difference as score of speed
        check_low = np.abs(self.post_condition[self.LOW] - result_state)
        check_high = np.abs(self.post_condition[self.HIGH] - result_state)

        score = np.array([check_low, check_high]).min(axis=0)

        center = (np.array(self.post_condition[self.LOW]) + np.array(self.post_condition[self.HIGH])) / 2
        # get score of angle
        if result_state[0] < center[0]:
            angle_diff = np.abs(center[0]-result_state[0])
        else:
            angle_diff = np.abs(result_state[0]-center[0])


        angle_score = (np.pi-angle_diff)/np.pi * 100

        speed_score = np.abs(center[1]-result_state[1])/self.post_condition[self.HIGH][1] * 100

        score[0] = angle_score
        score[1] = speed_score

        return score
