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
    def check_post_condition(self, result_state):
        check_low = np.abs(self.post_condition[self.LOW] - result_state)
        check_high = np.abs(self.post_condition[self.HIGH] - result_state)

        max_diff = np.array([check_low, check_high]).min(axis=0)

        return max_diff


def test_condition_processor(num_of_sampling_in_each_design):
    pre_condition_high = np.array([-np.pi / 2, 0.1])
    pre_condition_low = -pre_condition_high
    pre_condition = [pre_condition_low, pre_condition_high]
    post_condition = pre_condition

    output_sample_vector = []

    cond = ConditionProcessor(pre_condition, post_condition)
    for i in range(num_of_sampling_in_each_design):
        valid_new_state = cond.initial_state_generator()

        # evaluate process
        # There should be a evaluation function that return the result state
        # To test functionality of condition processor, we temporarily use random pre_condition to check

        max_diff = cond.check_post_condition(valid_new_state)

        output_sample_vector += [[valid_new_state, max_diff]]

    for i in output_sample_vector:
        print(i)
    return output_sample_vector


# test_condition_processor(10)
