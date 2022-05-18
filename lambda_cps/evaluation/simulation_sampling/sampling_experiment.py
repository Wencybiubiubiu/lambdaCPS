import os
from os.path import dirname, abspath
import numpy as np
import random
import lambda_cps
from lambda_cps.evaluation.simulation_sampling.sampler import Sampler
from lambda_cps.evaluation.fitting.linear_regression import LinearModel
from lambda_cps.evaluation.fitting.MLP_regression import MLPModel
from lambda_cps.evaluation.control.lqr import build_lqr_controller
from lambda_cps.envs import Pendulum
from lambda_cps.evaluation.reward.condition import ConditionProcessor


def flatten_the_samples(raw_data):
    new_data = []
    for i in range(len(raw_data)):
        cur_data = raw_data[i]
        design_data = np.array(cur_data[0])
        init_state_and_score_data = cur_data[1]

        for j in init_state_and_score_data:
            new_data.append([design_data.reshape(np.product(design_data.shape)), j[0], j[1]])

    return new_data

    # return raw_data.reshape(np.product(raw_data.shape[:-1]), -1)


def average_the_samples(raw_data):
    new_data = []
    for i in range(len(raw_data)):
        cur_data = raw_data[i]
        design_data = np.array(cur_data[0])
        init_state_and_score_data = np.array(cur_data[1])

        avg_init_state_and_score_data = np.mean(init_state_and_score_data, axis=0)
        new_data.append([design_data.reshape(np.product(design_data.shape)), avg_init_state_and_score_data[0],
                         avg_init_state_and_score_data[1]])

    return new_data

    # output = []
    # for i in raw_data:
    #     output.append(np.mean(i, axis=0))
    # return output


def get_mass_and_finalAngle_only(raw_data):
    output = []
    for i in raw_data:
        output.append([i[0][6], i[2][0]])

    return output


def process_data(data, mode):
    # angle_only_data = []
    # for i in data:
    #     temp = []
    #     for j in range(0, len(i[1])):
    #         temp.append([i[0], i[1][j][0][0], i[1][j][1][0]])
    #     angle_only_data.append(temp)
    #
    # angle_only_data = np.array(angle_only_data)

    output = None

    if mode == 'single_trajectory':
        output = flatten_the_samples(data)
        output = get_mass_and_finalAngle_only(output)
    elif mode == 'average':
        output = average_the_samples(data)
        output = get_mass_and_finalAngle_only(output)
    elif mode == 'average_and_only_mass_and_score':
        output = average_the_samples(data)
        output = get_mass_and_finalAngle_only(output)
    else:
        # default is single_trajectory mode
        output = flatten_the_samples(data)

    return output


def split_X_y(raw_data):
    X = []
    y = []
    for i in raw_data:
        X.append(i[:-1])
        y.append(i[-1])

    return X, y


def get_a_new_design():
    mass = random.uniform(0, 5)
    length = random.uniform(1, 10)

    connection_mat = [[0, 1], [1, 0]]
    feature_mat = [[0, 0], [mass, length]]
    return [connection_mat, feature_mat]


def get_lqr_controller():
    env = Pendulum()
    env.reset()
    Q = 100 * np.eye(2)
    R = 0.1 * np.eye(1)
    stable_state = np.array([0, 0])

    lqr_controller = build_lqr_controller(env, stable_state, Q, R)

    return env, lqr_controller


def set_new_design_to_env(input_env, input_design):
    new_env = input_env
    new_env.set_param('m', input_design[1][1][0])
    # env.set_param('l', input_design[1][1][1])
    return new_env


class SimulationSamplerExperiment:

    def __init__(self, data_mode, learning_model, num_of_designs, num_of_sample_for_each_design,
                 evaluation_time_for_each_sample, input_pre_condition, input_post_condition):

        self.experiment(data_mode, learning_model, num_of_designs, num_of_sample_for_each_design,
                        evaluation_time_for_each_sample, input_pre_condition, input_post_condition)

    # mode:
    # 1. single_trajectory: each trajectory of one-time simulation is a sample. A design may have several samples
    # 2: average: each design will only have one sample which is the average of its all trajectories
    # 3: average_and_only_mass_and_score: make the data one-to-one, [mass, score]

    def experiment(self, data_mode, learning_model, num_of_designs, num_of_sample_for_each_design,
                   evaluation_time_for_each_sample, input_pre_condition, input_post_condition):

        new_sampler = Sampler(num_of_sample_for_each_design, evaluation_time_for_each_sample)

        simulation_output = []

        cond = ConditionProcessor(input_pre_condition, input_post_condition)

        for i in range(num_of_designs):
            env, lqr_controller = get_lqr_controller()
            current_design = get_a_new_design()
            env = set_new_design_to_env(env, current_design)
            sampling = new_sampler.sample_one_design(env, lqr_controller, cond, current_design)
            for each_sample_ind in range(len(sampling[1])):
                sampling[1][each_sample_ind][1] = cond.score_calculator(sampling[1][each_sample_ind][1])
            simulation_output.append(sampling)

            if i % 10 == 0:
                print('The ' + str(i + 1) + 'th design with mass ' + str(current_design) + ' finished.')

        test_sampling = simulation_output

        # To test, we just need angle information
        # process the data
        generated_samples = process_data(test_sampling, data_mode)

        x, y = split_X_y(generated_samples)

        print('num_of_designs', num_of_designs)
        print('num_of_sample_for_each_design', num_of_sample_for_each_design)
        print('evaluation_time_for_each_sample', evaluation_time_for_each_sample)

        image_dir = dirname(
            dirname(abspath(lambda_cps.__file__))) + '/data/res/' + 'sampling_image/' + learning_model + '/'
        os.makedirs(image_dir, exist_ok=True)

        if learning_model == 'LR':
            LinearModel(x, y, input_pre_condition, input_post_condition, image_dir).fit()
        elif learning_model == 'MLPR':
            MLPModel(x, y, input_pre_condition, input_post_condition, image_dir).fit()
        else:
            # default is linear regression model
            LinearModel(x, y, input_pre_condition, input_post_condition, image_dir).fit()
