import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class MLPModel:

    def __init__(self, X, y):
        self.reg = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.image_dir = 'image/MLPR/'

    def train(self):
        self.reg = MLPRegressor(random_state=1, max_iter=10000)
        errors = []
        for i in range(10000):
            self.reg.partial_fit(self.X_train, self.y_train)
            #print(self.reg.loss_)

    def get_score(self, x, y):
        return self.reg.score(x, y)

    def test(self, input_x):
        return self.reg.predict(np.array(input_x))

    def save_plot(self, x_data, y_data, y_predict, data_type):

        processed_x = np.array(x_data)[:, 0]
        y_predict = np.array(y_predict)
        y_data = np.array(y_data)

        plt.scatter(np.array(x_data)[:, 0], y_data, label='original', color="black")

        cubic_interploation_model = interp1d(processed_x, y_predict, kind="cubic")
        X_=np.linspace(processed_x.min(), processed_x.max(), 500)
        Y_=cubic_interploation_model(X_)

        plt.axis([-0.1, processed_x.max()*1.1, min(y_predict.min(), y_data.min())/1.1,
                  max(y_predict.max(), y_data.max())*1.1])
        plt.plot(X_, Y_, label='predicted', color="blue", linewidth=3)
        plt.xlabel('Mass of pendulum')
        plt.ylabel('Score')
        plt.legend()
        plt.title(data_type + ' data of MLP regression model')
        plt.savefig(self.image_dir + data_type + '_figure.png')
        plt.close()


    def fit(self):
        self.train()
        prediction_y_test = self.test(self.X_test)
        prediction_y_train = self.test(self.X_train)
        prediction_all = self.test(self.X)

        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(self.y_test, prediction_y_test))
        print("Loss: %.2f" % mean_squared_error(self.y_test, prediction_y_test))

        self.save_plot(self.X_train, self.y_train, prediction_y_train, 'Train')
        self.save_plot(self.X_test, self.y_test, prediction_y_test, 'Test')
        self.save_plot(self.X, self.y, prediction_all, 'All')