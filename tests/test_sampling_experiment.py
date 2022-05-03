import numpy as np
import time
from lambda_cps.evaluation.sampling.sampling_experiment import SamplerExperiment

# Data processing mode:
# 1. single_trajectory: each trajectory of one-time simulation is a sample. A design may have several samples
# 2. average: each design will only have one sample which is the average of its all trajectories
# 3. average_and_only_mass_and_score: make the data one-to-one, [mass, score]
# Learning model:
# 1. LR: linear regression model.
# 2. MLPR: multi-layer perceptron regression model


def test_sampling_exp_LR():
    st = time.time()

    processing_mode = 'average_and_only_mass_and_score'
    learning_model_mode = 'LR'
    num_of_designs = 100
    num_of_simulations_for_each_design = 20
    num_of_steps_for_each_design = 100
    pre_condition = [[-np.pi / 2, 0.1], [np.pi / 2, 0.1]]
    post_condition = [[-np.pi / 4, -np.inf], [np.pi / 4, np.inf]]
    SamplerExperiment(processing_mode, learning_model_mode, num_of_designs, num_of_simulations_for_each_design,
                      num_of_steps_for_each_design, pre_condition, post_condition)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')



def test_sampling_exp_MLPR():
    st = time.time()

    processing_mode = 'average_and_only_mass_and_score'
    learning_model_mode = 'MLPR'
    num_of_designs = 100
    num_of_simulations_for_each_design = 20
    num_of_steps_for_each_design = 100
    pre_condition = [[-np.pi / 2, 0.1], [np.pi / 2, 0.1]]
    post_condition = [[-np.pi / 4, -np.inf], [np.pi / 4, np.inf]]
    SamplerExperiment(processing_mode, learning_model_mode, num_of_designs, num_of_simulations_for_each_design,
                      num_of_steps_for_each_design, pre_condition, post_condition)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')



if __name__ == '__main__':
    test_sampling_exp_LR()
    test_sampling_exp_MLPR()
