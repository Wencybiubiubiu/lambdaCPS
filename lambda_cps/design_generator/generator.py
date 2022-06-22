import random
from lambda_cps.parsing.parser import RuleParam
import pydot
from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import dirname, abspath
import os
import lambda_cps


class DesignGenerator(RuleParam):

    def __init__(self, production_rules_dict):
        super().__init__()
        self.production_rules_dict = production_rules_dict
        self.node_to_ = 'node_to_'
        self.edge_to_both = 'edge_to_both'
        self.edge_to_child = 'edge_to_child'
        self.edge_from_parent = 'edge_from_parent'

        self.image_dir = dirname(
            dirname(abspath(lambda_cps.__file__))) + '/data/res/' + 'generating_process/' + 'debugging' + '/'
        # os.makedirs(self.image_dir, exist_ok=True)

        self.execution_data_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S").replace('/', '').replace(':', '') \
            .replace(' ', '').replace(',', '_')
        self.new_folder = self.image_dir + self.execution_data_time + '/'


    def generate_networkx_image(self, input_graph, filename):

        pink_color = '#F2A7C6'
        blue_color = '#0E76D2'
        node_size = 1600
        attr_text_offset = 0.08
        margin_offset = 5

        my_networkx_graph = nx.drawing.nx_pydot.from_pydot(input_graph)
        labels = nx.get_node_attributes(my_networkx_graph, self.label)
        pos_nodes = nx.spring_layout(my_networkx_graph)
        nx.draw(my_networkx_graph, pos_nodes, with_labels=True, node_color=pink_color, node_size=node_size)

        pos_attrs = {}
        for node, coords in pos_nodes.items():
            pos_attrs[node] = (coords[0], coords[1] + attr_text_offset)

        custom_node_attrs = {}
        for node, attr in labels.items():
            custom_node_attrs[node] = attr

        nx.draw_networkx_labels(my_networkx_graph, pos_attrs, labels=custom_node_attrs, font_color=blue_color)
        plt.plot()
        plt.subplots_adjust(left=margin_offset + 0.1, right=margin_offset + 0.9, top=margin_offset + 0.6,
                             bottom=margin_offset + 0.4)
        ax1 = plt.subplot(111)
        ax1.margins(0.3)
        plt.savefig(filename + ".png")
        plt.close()

        return

    def get_all_possible_next_rules(self, prev_graph):

        # graph_info = prev_graph[self.RIGHT_GRAPH_TAG]
        # prev_graph_edge_list = prev_graph[self.RIGHT_RULE_EDGE_TAG]
        # prev_graph_node_list = prev_graph[self.RIGHT_RULE_NODE_TAG]

        prev_graph_edge_list = prev_graph.get_edge_list()
        prev_graph_node_list = prev_graph.get_node_list()

        # print(prev_graph)
        # exit()

        node_name_to_label_dict = {}
        for each_node in prev_graph_node_list:
            node_name_to_label_dict[each_node.get_name()] = each_node.get_attributes()[self.label]

        # print(node_name_to_label_dict)


        possible_actions = []

        for cur_rule_name in self.production_rules_dict:
            cur_rule = self.production_rules_dict[cur_rule_name]

            orig_edge_list = cur_rule[self.LEFT_RULE_EDGE_TAG]
            orig_node_list = cur_rule[self.LEFT_RULE_NODE_TAG]

            if len(orig_edge_list) == 0:
                replaced_node = orig_node_list[0]

                # print(cur_rule_name, replaced_node)

                for i in range(len(prev_graph_node_list)):
                    target_node = prev_graph_node_list[i]
                    target_node_label = target_node.get_attributes()[self.label].replace('\'','').replace('\"','')
                    replaced_node_label = replaced_node.get_attributes()[self.require_label].replace('\'','').replace('\"','')
                    # print(target_node.get_attributes())
                    if target_node_label == replaced_node_label:
                        possible_actions.append([self.node_to_, cur_rule_name, target_node.get_name()])
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
                                possible_actions.append([self.edge_to_both, cur_rule_name, target_edge.get_source(), target_edge.get_destination()])
                            elif replaced_edge_source == self.parent_node and \
                                    replaced_edge_des == node_name_to_label_dict[target_edge.get_destination()]:
                                possible_actions.append([self.edge_to_child, cur_rule_name, target_edge.get_source(), target_edge.get_destination()])
                            elif replaced_edge_des == self.child_node and \
                                    replaced_edge_source == node_name_to_label_dict[target_edge.get_source()]:
                                possible_actions.append([self.edge_from_parent, cur_rule_name, target_edge.get_source(), target_edge.get_destination()])
                    else:
                        # print(graph_info)
                        # print(graph_info.get_node("body"))
                        # print(node_name_to_label_dict)
                        # print(target_edge.get_destination(), node_name_to_label_dict[target_edge.get_destination()])
                        if replaced_edge_source == self.parent_node and replaced_edge_des == self.child_node:
                            possible_actions.append([self.edge_to_both, cur_rule_name, target_edge.get_source(), target_edge.get_destination()])
                        elif replaced_edge_source == self.parent_node and \
                                replaced_edge_des == node_name_to_label_dict[target_edge.get_destination()]:
                            possible_actions.append([self.edge_to_child, cur_rule_name, target_edge.get_source(), target_edge.get_destination()])
                        elif replaced_edge_des == self.child_node and \
                                replaced_edge_source == node_name_to_label_dict[target_edge.get_source()]:
                            possible_actions.append([self.edge_from_parent, cur_rule_name, target_edge.get_source(), target_edge.get_destination()])

        print('possible_actions', possible_actions)
        # exit()

        return node_name_to_label_dict, possible_actions

    def pick_action(self, possible_action_list):

        # print(possible_action_list[1])
        # return possible_action_list[1]

        return possible_action_list[random.randint(0, len(possible_action_list)-1)]

    def take_action(self, prev_graph, node_name_dict, picked_action, node_count):

        cur_graph = prev_graph
        temp_orig_name_2_new_name_dict = {}
        cur_node_count = node_count

        print(prev_graph)
        print('picked_action', picked_action)

        # exit()

        edit_method = picked_action[0]
        edit_rule = picked_action[1]
        right_hand_side_replacement = self.production_rules_dict[edit_rule][self.RIGHT_GRAPH_TAG]
        rhs_graph_edge_list = self.production_rules_dict[edit_rule][self.RIGHT_RULE_EDGE_TAG]
        lhs_graph_edge_list = self.production_rules_dict[edit_rule][self.LEFT_RULE_EDGE_TAG]
        rhs_graph_node_list = self.production_rules_dict[edit_rule][self.RIGHT_RULE_NODE_TAG]



        if edit_method != self.node_to_:

            for i in rhs_graph_node_list:
                # graph_info.add_node(i)
                # print(i)
                # print(node_name_dict)
                orig_name = i.get_name()
                if orig_name != 'child' and orig_name != 'parent':
                    orig_label = i.get_attributes()[self.label].replace('\"', '').replace('\'', '')
                    new_name = 'node_' + str(cur_node_count)
                    # print(new_name,cur_node_count)
                    cur_node_count = cur_node_count + 1
                    cur_graph.add_node(pydot.Node(new_name, label=orig_label))
                    node_name_dict[new_name] = orig_name
                    temp_orig_name_2_new_name_dict[orig_name] = new_name

                # print(node_name_dict, new_name, orig_name, orig_label)

        # exit()

        # print(node_name_dict)

            # This is the hard code to handle terminal node
            # Notice: It should be elaborated in the future !!!!
            # ????????
            if len(rhs_graph_edge_list) > 0:
                cur_graph.del_edge(picked_action[2], picked_action[3])

                if edit_method == self.edge_to_both:
                    cur_graph.del_node(picked_action[2])
                    cur_graph.del_node(picked_action[3])
                elif edit_method == self.edge_to_child:
                    cur_graph.del_node(picked_action[3])
                elif edit_method == self.edge_from_parent:
                    cur_graph.del_node(picked_action[2])

            for i in rhs_graph_edge_list:
                orig_edge_source = i.get_source()
                orig_edge_des = i.get_destination()
                edge_attr = i.get_attributes()
                edge_label = ''
                if self.label in edge_attr:
                    edge_label = edge_attr[self.label].replace('\"', '').replace('\'', '')

                if orig_edge_source == 'parent' and orig_edge_des == 'child':
                    orig_edge_source = picked_action[2]
                    orig_edge_des = picked_action[3]
                elif orig_edge_source == 'parent':
                    orig_edge_source = picked_action[2]
                    orig_edge_des = temp_orig_name_2_new_name_dict[orig_edge_des]
                elif orig_edge_des == 'child':
                    orig_edge_source = temp_orig_name_2_new_name_dict[orig_edge_source]
                    orig_edge_des = picked_action[3]
                else:
                    orig_edge_source = temp_orig_name_2_new_name_dict[orig_edge_source]
                    orig_edge_des = temp_orig_name_2_new_name_dict[orig_edge_des]

                cur_graph.add_edge(pydot.Edge(orig_edge_source, orig_edge_des, label=edge_label))

        else:

            fixed_node_name = picked_action[2]
            fixed_node_label = cur_graph.get_node(fixed_node_name)[0].get_attributes()[self.label]\
                .replace('\"', '').replace('\'', '')
            for i in rhs_graph_node_list:

                orig_name = i.get_name()
                if orig_name != 'child' and orig_name != 'parent':
                    # print(i.get_attributes())
                    orig_label = i.get_attributes()[self.label].replace('\"', '').replace('\'', '')
                    if orig_label != fixed_node_label:
                        new_name = 'node_' + str(cur_node_count)
                        # print(new_name,cur_node_count)
                        cur_node_count = cur_node_count + 1
                        cur_graph.add_node(pydot.Node(new_name, label=orig_label))
                        node_name_dict[new_name] = orig_name
                        temp_orig_name_2_new_name_dict[orig_name] = new_name
                    else:
                        node_name_dict[fixed_node_name] = orig_name
                        temp_orig_name_2_new_name_dict[orig_name] = fixed_node_name

            for i in rhs_graph_edge_list:
                orig_edge_source = i.get_source()
                orig_edge_des = i.get_destination()
                edge_attr = i.get_attributes()
                edge_label = ''
                if self.label in edge_attr:
                    edge_label = edge_attr[self.label].replace('\"', '').replace('\'', '')

                orig_edge_source = temp_orig_name_2_new_name_dict[orig_edge_source]
                orig_edge_des = temp_orig_name_2_new_name_dict[orig_edge_des]

                cur_graph.add_edge(pydot.Edge(orig_edge_source, orig_edge_des, label=edge_label))

        print(cur_graph)

        return cur_graph, node_name_dict, cur_node_count

    def get_a_new_design_with_max_steps(self, input_graph, max_steps):

        cur_graph = input_graph

        self.generate_networkx_image(cur_graph,"1")

        node_count = len(input_graph.get_node_list())
        for i in range(max_steps):
            name_dict, action_list = self.get_all_possible_next_rules(cur_graph)
            next_step = self.pick_action(action_list)
            cur_graph, name_dict, node_count = self.take_action(cur_graph, name_dict, next_step, node_count)

        # print(cur_graph)
        self.generate_networkx_image(cur_graph, "2")
        return cur_graph


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
