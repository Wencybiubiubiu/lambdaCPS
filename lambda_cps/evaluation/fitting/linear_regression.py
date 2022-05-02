import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class LinearModel:

    def __init__(self, X, y):
        self.reg = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.image_dir = 'image/LR/'

    def train(self):
        self.reg = LinearRegression().fit(self.X_train, self.y_train)

    def get_score(self, x, y):
        return self.reg.score(x, y)

    def test(self, input_x):
        # predict_result = []
        # for i in range(len(self.X_test)):
        #     predict_result.append(self.reg.predict(np.array(self.X_test[i])))
        # return predict_result

        return self.reg.predict(np.array(input_x))

    def save_plot(self, x_data, y_data, y_predict, filename):

        plt.scatter(np.array(x_data)[:, 0], y_data, color="black")
        plt.plot(np.array(x_data)[:, 0], y_predict, color="blue", linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.savefig(self.image_dir + filename + '.png')
        plt.close()


    def fit(self):
        self.train()
        prediction_y_test = self.test(self.X_test)
        prediction_y_train = self.test(self.X_train)
        prediction_all = self.test(self.X)

        # print(self.y_test, prediction_y_test)
        # print(self.get_score(self.X_train, self.y_train))
        # print(self.get_score(self.X_test, self.y_test))

        # The coefficients
        print("Coefficients: \n", self.reg.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(self.y_test, prediction_y_test))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(self.y_test, prediction_y_test))

        self.save_plot(self.X_train, self.y_train, prediction_y_train, 'train_figure')
        self.save_plot(self.X_test, self.y_test, prediction_y_test, 'test_figure')
        self.save_plot(self.X, self.y, prediction_all, 'all_figure')