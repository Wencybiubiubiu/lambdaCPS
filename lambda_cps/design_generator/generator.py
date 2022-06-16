import random
from lambda_cps.parsing.parser import RuleParam
import pydot
from graphviz import Digraph


class DesignGenerator(RuleParam):

    def __init__(self, production_rules_dict):
        super().__init__()
        self.production_rules_dict = production_rules_dict
        self.node_to_ = 'node_to_'
        self.edge_to_ = 'edge_to_'

    def get_all_possible_next_rules(self, prev_graph):

        graph_info = prev_graph[self.RIGHT_GRAPH_TAG]
        prev_graph_edge_list = prev_graph[self.RIGHT_RULE_EDGE_TAG]
        prev_graph_node_list = prev_graph[self.RIGHT_RULE_NODE_TAG]

        node_name_to_label_dict = {}
        for each_node in prev_graph_node_list:
            node_name_to_label_dict[each_node.get_name()] = each_node.get_attributes()[self.label]

        print(node_name_to_label_dict)


        possible_actions = []

        for key in self.production_rules_dict:
            cur_rule = self.production_rules_dict[key]

            orig_edge_list = cur_rule[self.LEFT_RULE_EDGE_TAG]
            orig_node_list = cur_rule[self.LEFT_RULE_NODE_TAG]

            if len(orig_edge_list) == 0:
                # print(key, orig_node_list)
                replaced_node = orig_node_list[0]

                for i in range(len(prev_graph_node_list)):
                    target_node = prev_graph_node_list[i]
                    if target_node.get_attributes()[self.label] == replaced_node.get_attributes()[self.require_label]:
                        possible_actions.append([self.node_to_, key, target_node.get_name()])
                        # print(output[-1])

            else:
                replaced_edge = orig_edge_list[0]
                replaced_edge_source = replaced_edge.get_source()
                replaced_edge_des = replaced_edge.get_destination()

                for i in range(len(prev_graph_edge_list)):
                    target_edge = prev_graph_edge_list[i]
                    # print(replaced_edge, target_edge)
                    if self.require_label in replaced_edge.get_attributes():
                        if target_edge.get_attributes()[self.label] == replaced_edge.get_attributes()[self.require_label]:
                            # print(replaced_edge_source, replaced_edge_des)
                            if replaced_edge_source == self.parent_node and replaced_edge_des == self.child_node:
                                possible_actions.append([self.edge_to_, key, target_edge.get_source(), target_edge.get_destination()])
                            elif replaced_edge_source == self.parent_node and \
                                    replaced_edge_des == node_name_to_label_dict[target_edge.get_destination()]:
                                possible_actions.append([self.edge_to_, key, any, target_edge.get_destination()])
                            elif replaced_edge_des == self.child_node and \
                                    replaced_edge_source == node_name_to_label_dict[target_edge.get_source()]:
                                possible_actions.append([self.edge_to_, key, target_edge.get_source(), any])
                    else:
                        # print(graph_info)
                        # print(graph_info.get_node("body"))
                        # print(target_edge.get_destination(), node_name_to_label_dict[target_edge.get_destination()])
                        if replaced_edge_source == self.parent_node and replaced_edge_des == self.child_node:
                            possible_actions.append([self.edge_to_, key, target_edge.get_source(), target_edge.get_destination()])
                        elif replaced_edge_source == self.parent_node and \
                                replaced_edge_des == node_name_to_label_dict[target_edge.get_destination()]:
                            possible_actions.append([self.edge_to_, key, any, target_edge.get_destination()])
                        elif replaced_edge_des == self.child_node and \
                                replaced_edge_source == node_name_to_label_dict[target_edge.get_source()]:
                            possible_actions.append([self.edge_to_, key, target_edge.get_source(), any])

        print(possible_actions)

        return node_name_to_label_dict, possible_actions

    def pick_action(self, possible_action_list):

        return possible_action_list[random.randint(0, len(possible_action_list)-1)]


    def take_action(self, prev_graph, node_name_dict, picked_action):

        graph_info = prev_graph[self.RIGHT_GRAPH_TAG]

        print(picked_action)
        edit_method = picked_action[0]
        edit_rule = picked_action[1]
        right_hand_side_replacement = self.production_rules_dict[edit_rule][self.RIGHT_GRAPH_TAG]
        rhs_graph_edge_list = self.production_rules_dict[edit_rule][self.RIGHT_RULE_EDGE_TAG]
        rhs_graph_node_list = self.production_rules_dict[edit_rule][self.RIGHT_RULE_NODE_TAG]

        for i in rhs_graph_node_list:
            graph_info.add_node(i)

        for i in rhs_graph_edge_list:
            graph_info.add_edge(i)

        print(graph_info)
        return

    def get_all_possible_next_rules_old(self, prev_graph, prev_rule_list):

        graph_info = prev_graph[self.RIGHT_RULE_TAG]
        prev_graph_edge_list = list(graph_info[self.EDGE_TAG])
        prev_graph_node_list = list(graph_info[self.NODE_TAG])

        print(prev_graph_edge_list)
        print(prev_graph_node_list)

        # format will be [rule_selected, parent_node_index_to_replace, child_node_index_to_replace,
        # edge_index_to_replace] if there is no edge or node, use -1 as index.
        output = []

        for key in self.production_rules_dict:
            cur_rule = self.production_rules_dict[key]
            orig_graph = cur_rule[self.LEFT_RULE_TAG]
            orig_edge_list = list(orig_graph[self.EDGE_TAG])
            orig_node_list = list(orig_graph[self.NODE_TAG])

            if len(orig_edge_list) == 0:

                print(orig_graph)
                cur_node = orig_node_list[0]
                node_label = orig_graph[self.ADJ_TAG]
                print(node_label)

                for i in range(len(prev_graph_node_list)):
                    if prev_graph_node_list[i] == cur_node:
                        output.append([key, i, -1, -1])

            # print(key, cur_rule)

        return

    # The functions outside the pipeline class should be elaborated in the future (wrapped in class)
    def get_a_new_design(self, learned_model, init_sketch, init_xml_file):
        mass = random.uniform(0, 5)
        length = random.uniform(1, 10)

        connection_mat = [[0, 1], [1, 0]]
        feature_mat = [[0, 0], [mass, length]]

        design_matrix = [connection_mat, feature_mat]
        new_xml_file = init_xml_file  # It should be updated instead of using initial one
        ancestors = None  # It should be all designs in the generating process of the target complete design
        return design_matrix, new_xml_file, ancestors
