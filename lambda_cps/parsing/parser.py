import pydot
import networkx
import graphviz


class RuleParam:

    def __init__(self):
        self.LEFT_RULE_INDEX = 0
        self.RIGHT_RULE_INDEX = 1

        self.ORIG_RULE_GRAPH_TAG = 'OG'
        self.LEFT_GRAPH_TAG = 'LF'
        self.RIGHT_GRAPH_TAG = 'RG'
        self.LEFT_RULE_NODE_TAG = 'LRN'
        self.RIGHT_RULE_NODE_TAG = 'RRN'
        self.LEFT_RULE_EDGE_TAG = 'LRE'
        self.RIGHT_RULE_EDGE_TAG = 'RRE'
        self.ADJ_TAG = 'A'
        self.NODE_TAG = 'N'
        self.EDGE_TAG = 'E'

        self.init_sketch_tag = 'make_robot'

        self.parent_node = 'parent'
        self.child_node = 'child'

        self.require_label = 'require_label'
        self.label = 'label'
        self.node_count = 'count'

    def create_new_node_name(self, input_number):
        return 'n' + str(input_number)

class Parser(RuleParam):

    def __init__(self, input_dot_file):
        super().__init__()
        self.rule_set_of_dot_format = pydot.graph_from_dot_file(input_dot_file)

        # a = self.rule_set_of_dot_format[0].get_subgraph('L')[0].get_node('robot')[0]
        # print(a.get_attributes())

        self.rule_dict = {}
        newline_node = self.rule_set_of_dot_format[0].get_subgraph('R')[0].get_node_list()[-1].get_name()
        for each_rule in self.rule_set_of_dot_format:
            rule_name = each_rule.get_name()

            left_rule = each_rule.get_subgraph('L')[0]
            left_rule.del_node(newline_node)
            left_rule_node_list = left_rule.get_node_list()
            left_rule_edge_list = left_rule.get_edge_list()

            right_rule = each_rule.get_subgraph('R')[0]
            right_rule.del_node(newline_node)
            right_rule_node_list = right_rule.get_node_list()
            right_rule_edge_list = right_rule.get_edge_list()


            self.rule_dict[rule_name] = {self.ORIG_RULE_GRAPH_TAG: each_rule,
                                         self.LEFT_GRAPH_TAG: left_rule, self.RIGHT_GRAPH_TAG: right_rule,
                                         self.LEFT_RULE_NODE_TAG: left_rule_node_list,
                                         self.LEFT_RULE_EDGE_TAG: left_rule_edge_list,
                                         self.RIGHT_RULE_NODE_TAG: right_rule_node_list,
                                         self.RIGHT_RULE_EDGE_TAG: right_rule_edge_list}

        # print(self.rule_dict)

    def __init__old(self, input_dot_file):
        super().__init__()
        self.rule_set_of_dot_format = pydot.graph_from_dot_file(input_dot_file)

        self.rule_dict = {}
        for each_rule in self.rule_set_of_dot_format:
            cur_networkx_graph = networkx.drawing.nx_pydot.from_pydot(each_rule)
            left_rule = networkx.drawing.nx_pydot.from_pydot(each_rule.get_subgraphs()[self.LEFT_RULE_INDEX])
            right_rule = networkx.drawing.nx_pydot.from_pydot(each_rule.get_subgraphs()[self.RIGHT_RULE_INDEX])

            self.rule_dict[cur_networkx_graph.name] = {}

            left_dict = {self.ADJ_TAG: left_rule.adj, self.NODE_TAG: left_rule.nodes, self.EDGE_TAG: left_rule.edges}
            right_dict = {self.ADJ_TAG: right_rule.adj, self.NODE_TAG: right_rule.nodes, self.EDGE_TAG: right_rule.edges}

            self.rule_dict[cur_networkx_graph.name] = {self.LEFT_GRAPH_TAG: left_rule, self.RIGHT_GRAPH_TAG: right_rule}
            #
            # print(each_rule)
            # print('cur_networkx_graph.name', cur_networkx_graph.name)
            print('left_rule.adj', left_rule.adj)
            print('left_rule.nodes', left_rule.nodes)
            print('left_rule.edges', left_rule.edges)
            # print('right_rule.adj', right_rule.adj)
            # print('right_rule.nodes', right_rule.nodes)
            # print('right_rule.edges', right_rule.edges)
            exit()

        # print(self.rule_dict)


    def get_rule_dict(self):

        return self.rule_dict

    def get_empty_sketch(self, init_sketch):

        if init_sketch is None:
            empty_sketch = self.rule_dict[self.init_sketch_tag]
            right_side = empty_sketch[self.RIGHT_GRAPH_TAG]

            # print(right_side)

            new_design = pydot.Dot('new-design', graph_type='digraph')
            # new_design = networkx.Graph(name='new-design')

            replace_name_dict = {}

            for i in range(len(empty_sketch[self.RIGHT_RULE_NODE_TAG])):
                orig_name = empty_sketch[self.RIGHT_RULE_NODE_TAG][i].get_name()
                orig_label = right_side.get_node(orig_name)[0].get_attributes()[self.label].replace('\"', '')
                new_name = self.create_new_node_name(i)
                new_design.add_node(pydot.Node(new_name, label=orig_label))
                # new_design.add_node(new_name, label=orig_label)
                replace_name_dict[orig_name] = new_name

            # print(empty_sketch[self.RIGHT_RULE_EDGE_TAG][0])
            for i in range(len(empty_sketch[self.RIGHT_RULE_EDGE_TAG])):
                source = empty_sketch[self.RIGHT_RULE_EDGE_TAG][i].get_source()
                des = empty_sketch[self.RIGHT_RULE_EDGE_TAG][i].get_destination()
                edge_attr = empty_sketch[self.RIGHT_RULE_EDGE_TAG][i].get_attributes()
                edge_label = edge_attr[self.label].replace('\"', '')
                new_design.add_edge(pydot.Edge(replace_name_dict[source], replace_name_dict[des], label=edge_label))
                # new_design.add_edge(replace_name_dict[source], replace_name_dict[des], label=edge_label)


            # empty_sketch[self.RIGHT_RULE_EDGE_TAG] = right_side.get_edge_list()
            # print(empty_sketch[self.RIGHT_RULE_EDGE_TAG][0])
            # print(empty_sketch[self.RIGHT_RULE_EDGE_TAG][1])
            # exit()

            # return self.rule_dict[self.init_sketch_tag], [self.init_sketch_tag]

            return new_design, [self.init_sketch_tag]

        else:

            print("We do not implement generating a design from non-empty sketch. Program quits.")
            exit()

            return None

