import pydot
import networkx


class Parser:

    def __init__(self):

        graphs = pydot.graph_from_dot_file('/Users/wenxianzhang/Desktop/mujocoApp/lambdaCPS/lambda_cps/rules/reacher.dot')
        graph = graphs[0]
        for i in graphs:
            print('=============')
            my_networkx_graph = networkx.drawing.nx_pydot.from_pydot(i)
            print(i)
            print(my_networkx_graph)
            print(my_networkx_graph.nodes)

            test_1 = networkx.drawing.nx_pydot.from_pydot(i.get_subgraphs()[0])
            print(test_1.nodes)
            print(test_1.nodes.items())
            print(test_1.edges)
            print(test_1.adj)
            test_1 = networkx.drawing.nx_pydot.from_pydot(i.get_subgraphs()[1])
            print(test_1.nodes)
            print(test_1.nodes.items())
            print(test_1.nodes.data())
            print(test_1.edges)
            print(test_1.adj)

        return


if __name__ == '__main__':
    Parser()