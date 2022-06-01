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

# The functions outside the pipeline class should be elaborated in the future (wrapped in class)
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
    new_env.set_param('l', input_design[1][1][1])
    return new_env

class Pipeline:

    def __init__(self):
        return

    def execute(self):

        # parameters
        CNNT_MATRIX = 'A'
        FEAT_MATRIX = 'D'
        TRAJ_VECTOR = 'T'
        SCORE_TAG = 'score'

        processing_mode = 'average_and_only_mass_and_score'
        learning_model_mode = 'MLPR'
        num_of_designs = 5
        num_of_simulations_for_each_design = 10
        num_of_steps_for_each_design = 10

        # input block
        # It should have 
        # 1. controller, 
        # 2. pre/post condition to restrict init and final state
        # 3. environment (included in controller)
        # 4. production rules
        # 5. input program sketch (we assume as empty now)
        pre_condition = [[-np.pi / 2, 0.1], [np.pi / 2, 0.1]]
        post_condition = [[-np.pi / 4, 0], [np.pi / 4, 10]]
        cond = ConditionProcessor(pre_condition, post_condition)

        # some tools
        new_sampler = Sampler(num_of_simulations_for_each_design, num_of_steps_for_each_design)

        # Generating round (with a loop)
        
        for i in range(num_of_designs):
            # design generating block

            current_design = get_a_new_design() # In the future, it should be able to take production rules as input

            # env update: embed new design into the environment
            env, lqr_controller = get_lqr_controller() # It should be reset as the design is changed
            env = set_new_design_to_env(env, current_design)

            # sample block
            # It will generate a list of samples with the same design
            sampling = new_sampler.sample_one_design(env, lqr_controller, cond, current_design)

            # score calculating block
            cur_score_list = []
            for each_sample_ind in range(len(sampling[TRAJ_VECTOR])):
                matching_score = cond.score_calculator(sampling[TRAJ_VECTOR][each_sample_ind][1])
                cur_score_list.append(matching_score[0])
            sampling[SCORE_TAG] = np.mean(cur_score_list)
            print(sampling)
            


        # sample format pre-processing format


        # GNN block: Graph Neural Network update by this sample


        # Search-guided block:



        return


if __name__ == '__main__':
    Pipeline().execute()