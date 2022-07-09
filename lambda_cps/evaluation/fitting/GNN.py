import torch
import torch.nn.functional as F
from torch_geometric.data import Data
# from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, Linear
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from os.path import dirname, abspath
import os
import lambda_cps
from datetime import datetime
import networkx as nx


class ParamName:

    def __init__(self):
        # parameters
        self.CNNT_MATRIX = 'A'
        self.COO_MATRIX = 'E'
        self.FEAT_MATRIX = 'D'
        self.TRAJ_VECTOR = 'T'
        self.SCORE_TAG = 'score'
        self.STEP_TAG = 'num_of_steps'

        return


class GCNDataWrapper(ParamName):

    def __init__(self):
        super().__init__()

        self.image_dir = dirname(
            dirname(abspath(lambda_cps.__file__))) + '/data/res/' + 'sampling_image/' + 'GCN' + '/'
        os.makedirs(self.image_dir, exist_ok=True)

        self.execution_data_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S").replace('/', '').replace(':', '') \
            .replace(' ', '').replace(',', '_')
        self.new_folder = self.image_dir + self.execution_data_time + '/'

        return

    def collect_data(self, input_single_dict):
        new_dict = {self.SCORE_TAG: input_single_dict[self.SCORE_TAG],
                    self.CNNT_MATRIX: input_single_dict[self.CNNT_MATRIX],
                    self.FEAT_MATRIX: input_single_dict[self.FEAT_MATRIX]}
        return new_dict

    def get_coo_format(self, input_single_dict):

        connection_matrix = input_single_dict[self.CNNT_MATRIX]

        x_axis = []
        y_axis = []

        for i in range(len(connection_matrix)):
            for j in range(len(connection_matrix[i])):

                if connection_matrix[i][j] > 0:
                    x_axis.append(i)
                    y_axis.append(j)

        input_single_dict[self.COO_MATRIX] = [x_axis, y_axis]

        return input_single_dict

    def convert_x_edge_y_to_tensor_data(self, input_single_dict):

        edge_index = torch.tensor(input_single_dict[self.COO_MATRIX], dtype=torch.long)
        x = torch.tensor(input_single_dict[self.FEAT_MATRIX], dtype=torch.float)
        # y = torch.tensor([input_single_dict[self.SCORE_TAG]], dtype=torch.long)
        y = torch.tensor([input_single_dict[self.SCORE_TAG]], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, y=y)

    def process_one(self, input_data):

        cur_data = self.collect_data(input_data)
        cur_data = self.get_coo_format(cur_data)
        tensor_data = self.convert_x_edge_y_to_tensor_data(cur_data)

        return tensor_data

    def process_all(self, input_dataset):

        new_dataset = []
        for i in range(len(input_dataset)):

            cur_data = input_dataset[i]
            # cur_data = self.collect_data(cur_data)
            # cur_data = self.get_coo_format(cur_data)
            # tensor_data = self.convert_x_edge_y_to_tensor_data(cur_data)
            # new_dataset.append(tensor_data)
            new_dataset.append(self.process_one(cur_data))

        return new_dataset


    def split_data(self, input_dataset, portion):

        index = int(len(input_dataset) * portion)
        training_set = input_dataset[:index]
        test_set = input_dataset[index:]

        return training_set, test_set


    def save_plot(self, x_data, y_data, y_predict, data_type):

        processed_x = np.array(x_data)
        y_predict = np.array(y_predict)
        y_data = np.array(y_data)

        plt.scatter(np.array(x_data), y_data, label='Real label', color="black")

        cubic_interploation_model = interp1d(processed_x, y_predict, kind="cubic")
        X_=np.linspace(processed_x.min(), processed_x.max(), 500)
        Y_=cubic_interploation_model(X_)

        plt.axis([-0.1, processed_x.max()*1.1, min(y_predict.min(), y_data.min())/1.1,
                  max(y_predict.max(), y_data.max())*1.1])
        plt.plot(X_, Y_, label='Predicted label', color="blue", linewidth=3)
        plt.xlabel('Mass of pendulum')
        plt.ylabel('Score')
        plt.legend()
        plt.title(data_type + ' dataset')
        plt.savefig(self.new_folder + data_type + '_figure.png')
        plt.close()

    def save_plot_scatter(self, x_data, y_data, y_predict, data_type):

        processed_x = np.array(x_data)
        y_predict = np.array(y_predict)
        y_data = np.array(y_data)

        plt.scatter(processed_x, y_data, label='Real label', color="black")
        plt.scatter(processed_x, y_predict, label='Predicted label', color="blue")
        plt.xlabel('Mass of pendulum')
        plt.ylabel('Score')
        plt.legend()
        plt.title(data_type + ' dataset')
        plt.savefig(self.new_folder + data_type + '_figure.png')
        plt.close()


    def evaluate(self, x, y, pred_y, data_type):
        print("Image stored in the folder: ", self.execution_data_time)

        os.makedirs(self.image_dir + self.execution_data_time + '/', exist_ok=True)

        #self.save_plot(x, y, pred_y, data_type)
        self.save_plot_scatter(x, y, pred_y, data_type)

    def get_single_sampled_data_dict(self, input_networkx_graph, num_of_steps, input_score):

        node_list = input_networkx_graph.nodes()
        output = {self.CNNT_MATRIX: nx.to_numpy_array(input_networkx_graph, nodelist=node_list).tolist()}

        feature_matrix = []
        for node in node_list:
            cur_indicator = int(input_networkx_graph.nodes[node]['indicator'])
            feature_matrix.append([cur_indicator])
        # print(feature_matrix)
        # exit()
        #
        # temp_size = 2
        # temp_feature_mat = []
        # for i in range(len(output[self.CNNT_MATRIX])):
        #     temp_feature_mat.append(np.zeros(temp_size).tolist())
        #
        # output[self.FEAT_MATRIX] = temp_feature_mat

        output[self.FEAT_MATRIX] = feature_matrix

        output[self.STEP_TAG] = num_of_steps
        output[self.SCORE_TAG] = input_score

        return output

    def convert_networkx_to_tensor_dict(self, input_networkx_graph, num_of_steps, score):

        sampling = self.get_single_sampled_data_dict(input_networkx_graph, num_of_steps, score)
        cur_tensor_data = self.process_one(sampling)

        return cur_tensor_data


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = GCNConv(dataset.num_node_features, 32)
        # self.conv2 = GCNConv(32, dataset.num_classes)
        self.l1 = Linear(10, 1)
        # self.conv1 = GCNConv(2, 32)
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(x.shape)
        # print(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # print(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # print(x)
        x = torch.mean(x, dim=0)
        # print(x)

        x = self.l1(x)
        # x = F.log_softmax(x, dim=0)
        # print(x)
        # exit()
        return x.unsqueeze(0)


class GCNModel:

    def __init__(self, learning_rate, weight_decay_rate):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)

        self.model.train()

        return

    def update_model_with_single_sample(self, input_sample):

        data = input_sample.to(self.device)
        self.optimizer.zero_grad()
        out = self.model(data)
        # print(out.squeeze(0), data.y)
        loss = F.l1_loss(out.squeeze(0), data.y)
        loss.backward()
        self.optimizer.step()
        self.model.eval()

        return

    def training(self, iteration_times, training_set):

        for epoch in range(iteration_times):

            for i in range(len(training_set)):

                # if i % 100 == 0:
                #     print('The ' + str(i + 1) + 'th graph in epoch ' + str(epoch) + ' is finished.')
                data = training_set[i].to(self.device)

                self.optimizer.zero_grad()
                out = self.model(data)
                # loss = F.nll_loss(out, data.y)
                loss = F.l1_loss(out.squeeze(0), data.y)
                # print(out, data.y, loss)
                loss.backward()
                self.optimizer.step()

        self.model.eval()

        return

    def predict_one(self, input_data):
        return self.model(input_data)
        # return self.model(input_data).argmax(dim=1)

    def predict_all(self, input_dataset):

        correct = 0
        predict_output = torch.Tensor([])
        real_label = torch.Tensor([])

        for i in range(len(input_dataset)):

            data = input_dataset[i]

            cur_pred = self.predict_one(data)
            predict_output = torch.cat((predict_output, cur_pred), 0)
            real_label = torch.cat((real_label, data.y), 0)

            if cur_pred == data.y:
                correct += 1

        accuracy = int(correct) / int(len(input_dataset)) * 100

        return predict_output, real_label, accuracy

# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
# new_GCN = GCNModel(1e-4, 5e-4)
# new_GCN.training(2, dataset)
# prediction_tensor_list, acc = new_GCN.predict_all(dataset)
# print(f'Accuracy: {acc:.4f}')
