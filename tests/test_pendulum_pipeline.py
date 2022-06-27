import numpy as np
from lambda_cps.evaluation.control.controller import Controller
from lambda_cps.evaluation.reward.condition import ConditionProcessor
from lambda_cps.evaluation.simulation_sampling.sampler import Sampler
from lambda_cps.evaluation.fitting.GNN import ParamName, GCNDataWrapper, GCNModel
from lambda_cps.design_generator.generator import DesignGenerator
from lambda_cps.parsing.parser import Parser


class Pipeline(ParamName):

    def __init__(self):
        super().__init__()
        return

    def execute(self):

        # training and generating parameters
        num_of_designs = 3
        num_of_simulations_for_each_design = 4
        num_of_steps_for_each_design = 10
        training_epochs = 10
        training_lr = 1e-3  # 1e-5
        training_weight_decay = 5e-5  # 5e-6
        train_test_partition_portion = 0.75

        # Set an initial GCN model
        new_GCN = GCNModel(training_lr, training_weight_decay)

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

        # Elaborated in the future
        init_sketch = None
        init_xml_file = None
        rule_file = './lambda_cps/rules/reacher.dot'

        # some tools
        new_sampler = Sampler(num_of_simulations_for_each_design, num_of_steps_for_each_design)
        new_parser = Parser(rule_file)
        new_generator = DesignGenerator(new_parser.get_rule_dict())
        new_controller = Controller()

        # Uncomment this line if you want to see the visualization of generating process in data/res/generating_process folder
        # new_generator.set_process_saving_flag()

        # Generating round (with a loop)
        # Currently, it generates a list of designs and matching scores, and then they are used in GCN training.
        # We should co-update RL model and GCN model, so we need to change it then.
        sampling_dataset = []
        for i in range(num_of_designs):

            # design searching and generating block
            # In the future, it should be able to take production rules as input
            current_design, cur_xml_file, ancestors_of_cur_design = new_generator.get_a_new_design(GCNModel,
                                                                                                   init_sketch,
                                                                                                   init_xml_file)

            # env update: embed new design into the environment
            env, lqr_controller = new_controller.get_env_and_controller()  # controller should be reset as the design is changed
            env = new_controller.set_new_design_to_env(env, current_design, cur_xml_file)

            # sample block
            # It will generate a list of samples with the same design
            sampling = new_sampler.sample_one_design(env, lqr_controller, cond, current_design)

            # score calculating block
            cur_score_list = []
            for each_sample_ind in range(len(sampling[self.TRAJ_VECTOR])):
                matching_score = cond.score_calculator(sampling[self.TRAJ_VECTOR][each_sample_ind][1])
                cur_score_list.append(matching_score[0])
            sampling[self.SCORE_TAG] = np.mean(cur_score_list)

            sampling_dataset.append(sampling)

            if i % 10 == 0:
                print('The ' + str(i + 1) + 'th design with A and D ' + str(current_design) + ' finished.')

        # sample format pre-processing format

        data_wrapper = GCNDataWrapper()
        new_sampling_dataset = data_wrapper.process_all(sampling_dataset)
        training_set, test_set = data_wrapper.split_data(new_sampling_dataset, train_test_partition_portion)

        # GNN block: Graph Neural Network update by this sample

        # We need a mechanism to update GCN model in every round of generating a new design
        new_GCN.training(training_epochs, training_set)
        # test_prediction_tensor_list, test_real_label_tensor_list, acc = new_GCN.predict_all(test_set)
        # train_prediction_tensor_list, train_real_label_tensor_list, acc = new_GCN.predict_all(training_set)
        # all_prediction_tensor_list, all_real_label_tensor_list, acc = new_GCN.predict_all(new_sampling_dataset)

        return


if __name__ == '__main__':
    Pipeline().execute()
