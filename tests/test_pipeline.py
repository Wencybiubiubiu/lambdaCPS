import numpy as np
import random

import numpy as np
from lambda_cps.envs import Pendulum
from lambda_cps.evaluation.control.lqr import build_lqr_controller
from lambda_cps.evaluation.reward.condition import ConditionProcessor
from lambda_cps.evaluation.simulation_sampling.sampler import Sampler
from lambda_cps.evaluation.fitting.GNN import ParamName, GCNDataWrapper, GCN, GCNModel


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



class Pipeline(ParamName):

    def __init__(self):
        super().__init__()
        return

    def execute(self):

        # training parameters
        num_of_designs = 30
        num_of_simulations_for_each_design = 10
        num_of_steps_for_each_design = 10
        training_epochs = 30
        training_lr = 1e-3 # 1e-5
        training_weight_decay = 5e-5 # 5e-6

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

        sampling_dataset = []
        for i in range(num_of_designs):
            # design generating block

            current_design = get_a_new_design() # In the future, it should be able to take production rules as input

            # env update: embed new design into the environment
            env, lqr_controller = get_lqr_controller() # It should be reset as the design is changed
            env = set_new_design_to_env(env, current_design)

            # sample block
            # It will generate a list of samples with the same design
            sampling = new_sampler.sample_one_design(env, lqr_controller, cond, current_design)

            print(sampling)

            exit()

            # score calculating block
            cur_score_list = []
            for each_sample_ind in range(len(sampling[self.TRAJ_VECTOR])):
                matching_score = cond.score_calculator(sampling[self.TRAJ_VECTOR][each_sample_ind][1])
                cur_score_list.append(matching_score[0])
            sampling[self.SCORE_TAG] = np.mean(cur_score_list)

            # It temporarily divides classes into 0,1,2,3,4,5,6,7,8,9
            # sampling[self.SCORE_TAG] = int(round(int(sampling[self.SCORE_TAG]), -1)/10)

            sampling_dataset.append(sampling)

            if i % 10 == 0:
                print('The ' + str(i + 1) + 'th design with A and D ' + str(current_design) + ' finished.')


        # sample format pre-processing format

        data_wrapper = GCNDataWrapper()
        new_sampling_dataset = data_wrapper.process_all(sampling_dataset)
        partition_portion = 0.75
        training_set, test_set = data_wrapper.split_data(new_sampling_dataset, partition_portion)

        # GNN block: Graph Neural Network update by this sample

        new_GCN = GCNModel(training_lr, training_weight_decay)
        new_GCN.training(training_epochs, training_set)
        test_prediction_tensor_list, test_real_label_tensor_list, acc = new_GCN.predict_all(test_set)
        train_prediction_tensor_list, train_real_label_tensor_list, acc = new_GCN.predict_all(training_set)
        all_prediction_tensor_list, all_real_label_tensor_list, acc = new_GCN.predict_all(new_sampling_dataset)

        # GCN evaluation block:

        # print(f'Accuracy: {acc:.4f}%')
        mass = []
        length = []
        for i in range(len(new_sampling_dataset)):
            cur_feature_mat = new_sampling_dataset[i].x.tolist()
            mass.append(cur_feature_mat[1][0])
            length.append(cur_feature_mat[1][1])

        training_mass, test_mass = data_wrapper.split_data(mass, partition_portion)
        training_length, test_length = data_wrapper.split_data(length, partition_portion)

        data_wrapper.evaluate(test_mass, test_prediction_tensor_list.tolist(), test_real_label_tensor_list.tolist(),
                              'Test_mass')
        data_wrapper.evaluate(training_mass, train_prediction_tensor_list.tolist(), train_real_label_tensor_list.tolist(),
                              'Train_mass')
        data_wrapper.evaluate(mass, all_prediction_tensor_list.tolist(), all_real_label_tensor_list.tolist(),
                              'All_mass')

        data_wrapper.evaluate(test_length, test_prediction_tensor_list.tolist(), test_real_label_tensor_list.tolist(),
                              'Test_length')
        data_wrapper.evaluate(training_length, train_prediction_tensor_list.tolist(), train_real_label_tensor_list.tolist(),
                              'Train_length')
        data_wrapper.evaluate(length, all_prediction_tensor_list.tolist(), all_real_label_tensor_list.tolist(),
                              'All_length')

        # Search-guided block:



        return


if __name__ == '__main__':
    Pipeline().execute()