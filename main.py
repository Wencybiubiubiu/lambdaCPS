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
        max_generating_steps = 10
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

        init_graph, init_rule_list = new_parser.get_empty_sketch(init_sketch)

        # Uncomment this line if you want to see the visualization of generating process in data/res/generating_process folder
        # new_generator.set_process_saving_flag()

        # Generating round (with a loop)
        sampling_dataset = []
        for i in range(num_of_designs):

            # design searching and generating block

            # It will produce a list of incomplete programs in the generating process
            # [p_0, p_1, p_2, ..., p_n], p_0 is the init sketch, p_n is the finally generated design

            graph_format = 'networkx'
            # It is heavy of reloading everytime, I will refine it later.
            init_graph, init_rule_list = new_parser.get_empty_sketch(init_sketch)
            designs_from_generating_process = new_generator.get_a_new_design_with_max_steps(init_graph,
                                                                                            max_generating_steps,
                                                                                            graph_format)

            ancestors_of_complete_design = designs_from_generating_process[:-1]

            # To process networkx class object, refer to https://networkx.org/documentation/stable/reference/introduction.html
            complete_design = designs_from_generating_process[-1]

            print(complete_design)

            # controller and environment
            # env update: embed new design into the environment
            # controller may be reset as the design is changed
            env, controller = None, None

            # simulation block
            # It should get the trajectory of simulation (output) of the generated design
            trajectory = None

            # score calculating block
            score = 0

            # Some training procedure
            # GCN and RL

        return


if __name__ == '__main__':
    Pipeline().execute()
