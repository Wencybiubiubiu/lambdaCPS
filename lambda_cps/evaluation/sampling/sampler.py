# evaluator (number_of_samples_of_this_design, current_design, pre-condition theta, post-condition alpha)
#   = [s_1,s_2,s_3,...,s_n], n is the number of samples for current design

from lambda_cps.evaluation.sampling.condition import ConditionProcessor
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

    def get_lqr_controller(self):

        env = Pendulum()
        env.reset()
        Q = 100 * np.eye(2)
        R = 0.1 * np.eye(1)
        stable_state = np.array([0, 0])

        lqr_controller = build_lqr_controller(env, stable_state, Q, R)

        return env, lqr_controller

    def evaluate_one_design(self, env, lqr_controller, current_design, starting_state):

        env.set_param('m', current_design)
        env.set_state(starting_state)
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
        env, lqr_controller = self.get_lqr_controller()
        for i in range(self.num_of_sample_for_each_design):
            valid_new_state = cond.initial_state_generator()
            result_state = self.evaluate_one_design(env, lqr_controller, current_design, valid_new_state)
            max_diff = cond.check_post_condition(result_state)

            output_sample_vector += [[valid_new_state, max_diff]]

        return output_sample_vector

    def sample(self):

        output = []

        for i in range(self.num_of_designs):

            current_design = self.get_a_new_design()
            output.append([current_design, self.sample_one_design(current_design)])

            if i % 10 == 0:
                print('The ' + str(i+1) + 'th design with mass ' + str(current_design) + ' finished.')

        # The output format will be a list of matrices that
        # <mass, [valid_state,max_diff], [valid_state,max_diff],...>

        # print(output)
        return output

    def flatten_the_samples(self, raw_data):
        return raw_data.reshape(raw_data.shape[0] * raw_data.shape[1], -1)

    def average_the_samples(self, raw_data):
        output = []
        for i in raw_data:
            output.append(np.mean(i, axis=0))
        return output

    def get_mass_and_finalAngle_only(self, raw_data):

        output = []
        for i in raw_data:
            output.append([i[0], i[2]])

        return output

    # mode:
    # 1. single_trajectory: each trajectory of one-time simulation is a sample. A design may have several samples
    # 2: average: each design will only have one sample which is the average of its all trajectories
    # 3: average_and_only_mass_and_score: make the data one-to-one, [mass, score]
    def process_data(self, data, mode):

        angle_only_data = []
        for i in data:
            temp = []
            for j in range(0, len(i[1])):
                temp.append([i[0], i[1][j][0][0], i[1][j][1][0]])
            angle_only_data.append(temp)

        angle_only_data = np.array(angle_only_data)

        output = None
        if mode == 'single_trajectory':
            output = self.flatten_the_samples(angle_only_data)
        elif mode == 'average':
            output = self.average_the_samples(angle_only_data)
        elif mode == 'average_and_only_mass_and_score':
            output = self.average_the_samples(angle_only_data)
            output = self.get_mass_and_finalAngle_only(output)
        else:
            # default is single_trajectory mode
            output = self.flatten_the_samples(angle_only_data)

        return output
