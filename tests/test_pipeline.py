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


class Pipeline:

    def __init__(self):
        return

    def execute(self):

        # input block

        processing_mode = 'average_and_only_mass_and_score'
        learning_model_mode = 'MLPR'
        num_of_designs = 100
        num_of_simulations_for_each_design = 20
        num_of_steps_for_each_design = 100
        pre_condition = [[-np.pi / 2, 0.1], [np.pi / 2, 0.1]]
        post_condition = [[-np.pi / 4, -np.inf], [np.pi / 4, np.inf]]

        # Generating round (with a loop)

        # design generating block


        # env update: embed new design into the environment


        # sample block


        # score calculating block


        # sample format pre-processing format


        # GNN block: Graph Neural Network update by this sample


        # Search-guided block:



        return


if __name__ == '__main__':
    Pipeline().execute()