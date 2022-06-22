from lambda_cps.design_generator.generator import DesignGenerator
from lambda_cps.parsing.parser import Parser


if __name__ == '__main__':

    # rule_file = '/Users/wenxianzhang/Desktop/mujocoApp/lambdaCPS/lambda_cps/rules/reacher.dot'
    rule_file = '/Users/wenxianzhang/Desktop/mujocoApp/lambdaCPS/lambda_cps/rules/RoboGrammar.dot'
    new_parser = Parser(rule_file)
    rule_dict = new_parser.get_rule_dict()
    new_generator = DesignGenerator(rule_dict)
    init_graph, init_rule_list = new_parser.get_empty_sketch()

    # name_dict, action_list = new_generator.get_all_possible_next_rules(init_graph)
    # next_step = new_generator.pick_action(action_list)
    # new_generator.take_action(init_graph, name_dict, next_step)

    new_generator.get_a_new_design_with_max_steps(init_graph, 4)