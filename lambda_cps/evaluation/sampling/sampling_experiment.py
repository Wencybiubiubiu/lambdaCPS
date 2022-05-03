import os
from os.path import dirname, abspath
import lambda_cps
from lambda_cps.evaluation.sampling.sampler import Sampler
from lambda_cps.evaluation.fitting.linear_regression import LinearModel
from lambda_cps.evaluation.fitting.MLP_regression import MLPModel


class SamplerExperiment:

    def __init__(self, data_mode, learning_model, num_of_designs, num_of_sample_for_each_design,
                 evaluation_time_for_each_sample, input_pre_condition, input_post_condition):

        self.experiment(data_mode, learning_model, num_of_designs, num_of_sample_for_each_design,
                        evaluation_time_for_each_sample, input_pre_condition, input_post_condition)

    def split_X_y(self, raw_data):
        X = []
        y = []
        for i in raw_data:
            X.append(i[:-1])
            y.append(i[-1])
        return X, y

    def experiment(self, data_mode, learning_model, num_of_designs, num_of_sample_for_each_design,
                   evaluation_time_for_each_sample, input_pre_condition, input_post_condition):

        new_sampler = Sampler(input_pre_condition, input_post_condition, num_of_sample_for_each_design,
                              evaluation_time_for_each_sample, num_of_designs)
        test_sampling = new_sampler.sample()

        # To test, we just need angle information
        # process the data
        generated_samples = new_sampler.process_data(test_sampling, data_mode)

        x, y = self.split_X_y(generated_samples)

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

