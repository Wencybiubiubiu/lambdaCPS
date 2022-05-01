import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score


class LinearModel:

    def __init__(self, X, y):
        self.reg = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

        plt.scatter(np.array(self.X_train)[:, 0], self.y_train, color="black")
        plt.plot(np.array(self.X_train)[:, 0], prediction_y_train, color="blue", linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.savefig('image/train_figure.png')
        plt.close()

        plt.scatter(np.array(self.X_test)[:, 0], self.y_test, color="black")
        plt.plot(np.array(self.X_test)[:, 0], prediction_y_test, color="blue", linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.savefig('image/test_figure.png')
        plt.close()

        plt.scatter(np.array(self.X)[:, 0], self.y, color="black")
        plt.plot(np.array(self.X)[:, 0], prediction_all, color="blue", linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.savefig('image/all_figure.png')
        plt.close()