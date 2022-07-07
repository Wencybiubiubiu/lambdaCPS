import numpy as np
import random
import networkx as nx
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


    def reward(self, generated_design):

        return 100-len(generated_design.nodes())*10

    def execute(self):

        # training and generating parameters
        num_of_designs = 100
        max_generating_steps = 10
        num_of_simulations_for_each_design = 4
        num_of_steps_for_each_design = 10
        training_epochs = 10
        training_lr = 1e-3  # 1e-5
        training_weight_decay = 5e-5  # 5e-6
        train_test_partition_portion = 0.75

        decay_rate = 1
        decay_coefficient = 0.95
        decay_frequency = 10

        # Set an initial GCN model
        new_GCN = GCNModel(training_lr, training_weight_decay)

        # input block
        # It should have
        # 1. controller,
        # 2. pre/post condition to restrict init and final state
        # 3. environment (included in controller)
        # 4. production rules
        # 5. input program sketch (we assume as empty now)
        pre_condition = None
        post_condition = None
        init_sketch = None
        init_xml_file = None
        rule_file = './lambda_cps/rules/reacher.dot'

        # some tools
        new_sampler = Sampler(num_of_simulations_for_each_design, num_of_steps_for_each_design)
        new_parser = Parser(rule_file)
        new_generator = DesignGenerator(new_parser.get_rule_dict())
        new_controller = Controller()
        data_wrapper = GCNDataWrapper()
        new_GCN = GCNModel(training_lr, training_weight_decay)

        init_graph, init_rule_list = new_parser.get_empty_sketch(init_sketch)


        # Generating round (with a loop)
        sampling_dataset = []
        for i in range(num_of_designs):

            # design searching and generating block

            # It will produce a list of incomplete programs in the generating process
            # [p_0, p_1, p_2, ..., p_n], p_0 is the init sketch, p_n is the finally generated design

            graph_format = 'networkx'
            # It is heavy of reloading everytime, I will refine it later.
            init_graph, init_rule_list = new_parser.get_empty_sketch(init_sketch)

            if i == num_of_designs - 1:
                # Uncomment this line if you want to see the visualization of generating process in data/res/generating_process folder
                new_generator.set_process_saving_flag()

            # if i%(num_of_designs/decay_frequency) == 0:
            decay_rate = decay_rate * decay_coefficient
            # print(decay_rate)

            # for j in range(100):
            #     decay_rate = decay_rate * decay_coefficient
            #     print(j,decay_rate)
            # exit()

            designs_from_generating_process = new_generator.get_a_new_design_with_max_steps(new_GCN,
                                                                                            init_graph,
                                                                                            max_generating_steps,
                                                                                            decay_rate,
                                                                                            graph_format)

            # designs_from_generating_process = new_generator.get_a_new_design_with_max_steps(self.reward,
            #                                                                                 init_graph,
            #                                                                                 max_generating_steps,
            #                                                                                 decay_rate,
            #                                                                                 graph_format)

            ancestors_of_complete_design = designs_from_generating_process[:-1]

            # To process networkx class object, refer to https://networkx.org/documentation/stable/reference/introduction.html
            complete_design = designs_from_generating_process[-1]

            # print(complete_design)

            # controller and environment
            # env update: embed new design into the environment
            # controller may be reset as the design is changed
            env, controller = None, None

            # simulation block
            # It should get the trajectory of simulation (output) of the generated design
            trajectory = None

            # score calculating block
            cur_score = random.randint(0, 100)
            cur_score = self.reward(complete_design)
            # print(cur_score)
            # print(complete_design)

            cur_tensor_data = data_wrapper.convert_networkx_to_tensor_dict(complete_design, len(designs_from_generating_process), cur_score)
            sampling_dataset.append(cur_tensor_data)

            # exit()

            # Some training procedure
            # GCN and RL

            # # print(cur_tensor_data)
            # for j in range(training_epochs):
            #     new_GCN.update_model_with_single_sample(cur_tensor_data)

            new_GCN.training(training_epochs, sampling_dataset)

        return


if __name__ == '__main__':
    Pipeline().execute()
