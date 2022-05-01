import numpy as np
from lambda_cps.evaluation.sampling.sampler import Sampler
from lambda_cps.evaluation.fitting.linear_regression import LinearModel


def flatten_the_samples(raw_data):
    return raw_data.reshape(raw_data.shape[0]*raw_data.shape[1], -1)


def average_the_samples(raw_data):
    output = []
    for i in raw_data:
        output.append(np.mean(i, axis=0))
    return output


def split_X_y(raw_data):

    X = []
    y = []
    for i in raw_data:
        X.append(i[:-1])
        y.append(i[-1])
    return X, y


def experiment():

    pre_condition_high = np.array([-np.pi / 1, 0.1])
    pre_condition_low = pre_condition_high
    pre_condition = [pre_condition_low, pre_condition_high]
    post_condition_high = np.array([-np.pi / 2, 0.1])
    post_condition_low = -pre_condition_high
    post_condition = [post_condition_low, post_condition_high]

    num_of_sample_for_each_design = 5
    evaluation_time_for_each_sample = 100
    num_of_designs = 20

    test_sampling = Sampler(pre_condition, post_condition, num_of_sample_for_each_design,
                            evaluation_time_for_each_sample, num_of_designs).sample()

    # To test, we just need angle information
    # process the data

    angle_only_data = []
    for i in test_sampling:
        temp = []
        for j in range(0, len(i[1])):
            temp.append([i[0], i[1][j][0][0], i[1][j][1][0]])
        angle_only_data.append(temp)


    angle_only_data = np.array(angle_only_data)
    flattened_sample = flatten_the_samples(angle_only_data)
    # averaged_sample = average_the_samples(angle_only_data)

    x, y = split_X_y(flattened_sample)

    LinearModel(x, y).fit()


experiment()
