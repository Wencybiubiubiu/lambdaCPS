import numpy as np
import time
from lambda_cps.evaluation.sampling.sampler import Sampler
from lambda_cps.evaluation.fitting.linear_regression import LinearModel
from lambda_cps.evaluation.fitting.MLP_regression import MLPModel

def split_X_y(raw_data):
    X = []
    y = []
    for i in raw_data:
        X.append(i[:-1])
        y.append(i[-1])
    return X, y


def experiment(data_mode, learning_model, num_of_designs, num_of_sample_for_each_design,
               evaluation_time_for_each_sample):
    pre_condition_high = np.array([-np.pi / 2, 0.1])
    pre_condition_low = np.array([np.pi / 2, 0.1])
    pre_condition = [pre_condition_low, pre_condition_high]
    post_condition_high = np.array([-np.pi / 4, 0])
    post_condition_low = np.array([np.pi / 4, 1])
    post_condition = [post_condition_low, post_condition_high]

    new_sampler = Sampler(pre_condition, post_condition, num_of_sample_for_each_design,
                          evaluation_time_for_each_sample, num_of_designs)
    test_sampling = new_sampler.sample()

    # To test, we just need angle information
    # process the data
    generated_samples = new_sampler.process_data(test_sampling, data_mode)

    x, y = split_X_y(generated_samples)

    print('num_of_designs', num_of_designs)
    print('num_of_sample_for_each_design', num_of_sample_for_each_design)
    print('evaluation_time_for_each_sample', evaluation_time_for_each_sample)

    if learning_model == 'LR':
        LinearModel(x, y).fit()
    elif learning_model == 'MLPR':
        MLPModel(x, y).fit()
    else:
        # default is linear regression model
        LinearModel(x, y).fit()



st = time.time()

# Data processing mode:
# 1. single_trajectory: each trajectory of one-time simulation is a sample. A design may have several samples
# 2. average: each design will only have one sample which is the average of its all trajectories
# 3. average_and_only_mass_and_score: make the data one-to-one, [mass, score]
# Learning model:
# 1. LR: linear regression model.
# 2. MLPR: multi-layer perceptron regression model

processing_mode = 'average_and_only_mass_and_score'
learning_model_mode = 'MLPR'
numOfDesigns = 200
numOfSimulationsForEachDesign = 20
numOfStepsForEachDesign = 100
experiment(processing_mode, learning_model_mode, numOfDesigns, numOfSimulationsForEachDesign, numOfStepsForEachDesign)

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')