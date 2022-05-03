import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.interpolate import interp1d
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class MLPModel:

    def __init__(self, X, y, pre_condition, post_condition, image_dir):
        self.reg = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.image_dir = image_dir

        self.title_string = 'Pre: [' + str(round(pre_condition[0][0], 2)) + ' rad & ' + str(pre_condition[0][1]) + \
                            ' rad/s, ' + str(round(pre_condition[1][0], 2)) + ' rad & ' + str(pre_condition[1][1]) + \
                            ' rad/s]\n' + \
                            'Post: [' + str(round(post_condition[0][0], 2)) + ' rad & ' + str(post_condition[0][1]) + \
                            ' rad/s, ' + str(round(post_condition[1][0], 2)) + ' rad & ' + str(post_condition[1][1]) + \
                            ' rad/s]'

    def train(self):
        self.reg = MLPRegressor(random_state=1, max_iter=10000)
        errors = []
        for i in range(10000):
            self.reg.partial_fit(self.X_train, self.y_train)

    def get_score(self, x, y):
        return self.reg.score(x, y)

    def test(self, input_x):
        return self.reg.predict(np.array(input_x))

    def save_plot(self, x_data, y_data, y_predict, folder_name, data_type):

        processed_x = np.array(x_data)[:, 0]
        y_predict = np.array(y_predict)
        y_data = np.array(y_data)

        plt.scatter(np.array(x_data)[:, 0], y_data, label='Real label', color="black")

        cubic_interploation_model = interp1d(processed_x, y_predict, kind="cubic")
        X_=np.linspace(processed_x.min(), processed_x.max(), 500)
        Y_=cubic_interploation_model(X_)

        plt.axis([-0.1, processed_x.max()*1.1, min(y_predict.min(), y_data.min())/1.1,
                  max(y_predict.max(), y_data.max())*1.1])
        plt.plot(X_, Y_, label='Predicted label', color="blue", linewidth=3)
        plt.xlabel('Mass of pendulum')
        plt.ylabel('Score')
        plt.legend()
        plt.title(data_type + ' dataset \n' + self.title_string)
        plt.savefig(folder_name + data_type + '_figure.png')
        plt.close()


    def fit(self):
        self.train()
        prediction_y_test = self.test(self.X_test)
        prediction_y_train = self.test(self.X_train)
        prediction_all = self.test(self.X)

        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(self.y_test, prediction_y_test))
        print("Loss: %.2f" % self.reg.loss_)

        execution_data_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S").replace('/', '').replace(':', '')\
            .replace(' ', '').replace(',','_')
        new_folder = self.image_dir + execution_data_time + '/'
        print("Image stored in the folder: ", execution_data_time)

        os.makedirs(self.image_dir + execution_data_time + '/', exist_ok=True)

        self.save_plot(self.X_train, self.y_train, prediction_y_train, new_folder, 'Train')
        self.save_plot(self.X_test, self.y_test, prediction_y_test, new_folder, 'Test')
        self.save_plot(self.X, self.y, prediction_all, new_folder, 'All')