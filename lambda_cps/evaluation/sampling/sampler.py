# evaluator (number_of_samples_of_this_design, current_design, pre-condition theta, post-condition alpha)
#   = [s_1,s_2,s_3,...,s_n], n is the number of samples for current design

from condition import ConditionProcessor
import numpy as np
import random

from lambda_cps.evaluation.control.lqr import build_lqr_controller
from lambda_cps.envs import Pendulum


class Sampler:

    def __init__(self, pre_condition, post_condition, num_of_sample_for_each_design, evaluation_time_for_each_sample,
                 num_of_designs):
        self.pre_condition = pre_condition
        self.post_condition = post_condition
        self.num_of_sample_for_each_design = num_of_sample_for_each_design
        self.evaluation_time_for_each_sample = evaluation_time_for_each_sample
        self.num_of_designs = num_of_designs


    def get_a_new_design(self):

        return random.uniform(0, 3)

    def evaluate_one_design(self, current_design, starting_state):

        env = Pendulum(current_design)
        env.set_state(starting_state)

        Q = 100 * np.eye(2)
        R = 0.1 * np.eye(1)
        stable_state = np.array([0, 0])

        lqr_controller = build_lqr_controller(env, stable_state, Q, R)

        obs = starting_state
        rew_sum = 0
        for i in range(self.evaluation_time_for_each_sample):
            action = lqr_controller.predict(obs)

            obs, r, _, _ = env.step(action)
            rew_sum += r
            #env.render()

        result_state = env.get_state()

        return result_state

    def sample_one_design(self, current_design):

        output_sample_vector = []

        cond = ConditionProcessor(self.pre_condition, self.post_condition)
        for i in range(self.num_of_sample_for_each_design):
            valid_new_state = cond.initial_state_generator()
            result_state = self.evaluate_one_design(current_design, valid_new_state)
            max_diff = cond.check_post_condition(result_state)

            output_sample_vector += [[valid_new_state, max_diff]]

        return output_sample_vector

    def sample(self):

        output = []

        for i in range(self.num_of_designs):

            current_design = self.get_a_new_design()
            output.append([current_design, self.sample_one_design(current_design)])

        return output


def test_sampler():
    pre_condition_high = np.array([-np.pi / 1, 0.1])
    pre_condition_low = -pre_condition_high
    pre_condition = [pre_condition_low, pre_condition_high]
    post_condition_high = np.array([-np.pi / 2, 0.1])
    post_condition_low = -pre_condition_high
    post_condition = [post_condition_low, post_condition_high]

    num_of_sample_for_each_design = 2
    evaluation_time_for_each_sample = 100
    num_of_designs = 2

    # there should be a function call to get the new design in each round
    # We temporarily set it as 0.1 to test

    test_sampling = Sampler(pre_condition, post_condition, num_of_sample_for_each_design,
                            evaluation_time_for_each_sample, num_of_designs)
    print(test_sampling.sample())


# test_sampler()
