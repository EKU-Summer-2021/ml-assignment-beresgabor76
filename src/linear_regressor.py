"""
Module for Linear Regression class
"""
import os
import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Class for training a Linear Regression model, and then test it
    """
    def __init__(self):
        """
        Constructor creates a LinearRegression model
        """
        self.__lin_reg = LinearRegression()
        self.__test_data = None
        self.__test_set = None
        self.__target = None
        self.__prediction = None
        self.__parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '../results/linear_regression')

    def train(self, train_set_x, train_set_y):
        """
        Trains the LinearRegression model with passed data
        """
        self.__lin_reg.fit(train_set_x, train_set_y)

    def test(self, test_data, test_set_x, test_set_y):
        """
        Tests the LinearRegression model with passed data
        """
        self.__test_data = test_data.copy()
        self.__test_data.reset_index(inplace=True)
        self.__test_set = pd.DataFrame(test_set_x.copy())
        self.__test_set.reset_index(inplace=True)
        self.__target = pd.DataFrame(test_set_y)
        self.__target.reset_index(inplace=True)
        score = self.__lin_reg.score(test_set_x, test_set_y)
        print("\nScore for test set: " + str(score))
        self.__prediction = pd.DataFrame(self.__lin_reg.predict(test_set_x),
                                         index=test_set_x.index,
                                         columns=['prediction'])
        self.__prediction.reset_index(inplace=True)

    def plot_results(self):
        """
        Plots out how the predicted values approximate the real ones
        """
        fig = plt.figure()
        min_x = min_y = self.__target.min()
        max_x = max_y = self.__target.max()
        plt.plot([min_x, max_x], [min_y, max_y])
        plt.scatter(self.__target, self.__prediction, alpha=0.5)
        save_path = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), self.__parent_dir, save_path)):
            os.chdir(self.__parent_dir)
            os.mkdir(save_path)
        plot_file = os.path.join(os.path.dirname(__file__), self.__parent_dir, save_path, 'results.png')
        fig.savefig(plot_file)

    def save_results(self):
        """
        Saves test dataset with target values and prediction results with errors
        """
        results_df = pd.concat([self.__test_data,
                                self.__target,
                                self.__prediction], axis=1).drop(['index'], axis=1)
        results_df['error'] = results_df['prediction'] - results_df['charges']
        results_df['error_pc'] = (results_df['prediction'] - results_df['charges'])\
                                / results_df['charges'] * 100
        results_df = results_df.round(2)
        save_path = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), self.__parent_dir, save_path)):
            os.chdir(self.__parent_dir)
            os.mkdir(save_path)
        csv_file = os.path.join(os.path.dirname(__file__), self.__parent_dir, save_path, 'results.csv')
        results_df.to_csv(path_or_buf=csv_file, index=False)

