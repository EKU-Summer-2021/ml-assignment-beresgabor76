"""
Module for Linear Regression class
"""
import os
import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearRegressor:
    """
    Class for training a Linear Regression model, and then test it
    """

    def __init__(self, saving_strategy, plotting_strategy):
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
        self.__sub_dir = self.__make_save_dir()
        self.__saving_strategy = saving_strategy
        self.__plotting_strategy = plotting_strategy

    def __make_save_dir(self):
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        sub_dir = save_time.strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), self.__parent_dir, sub_dir)):
            os.chdir(self.__parent_dir)
            os.mkdir(sub_dir)
        return sub_dir

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
        self.__test_data = self.__test_data.drop('index', axis=1)
        self.__test_set = pd.DataFrame(test_set_x.copy())
        self.__test_set.reset_index(inplace=True)
        self.__test_set = self.__test_set.drop('index', axis=1)
        self.__target = pd.DataFrame(test_set_y)
        self.__target.reset_index(inplace=True)
        self.__target = self.__target.drop('index', axis=1)
        score = self.__lin_reg.score(test_set_x, test_set_y)
        print("\nScore for test set: " + str(score))
        self.__prediction = pd.DataFrame(self.__lin_reg.predict(test_set_x),
                                         index=test_set_x.index,
                                         columns=['prediction'])
        self.__prediction.reset_index(inplace=True)
        self.__prediction = self.__prediction.drop('index', axis=1)

    def plot_results(self):
        """
        Plots out how the predicted values approximate the real ones
        """
        self.__plotting_strategy.plot_results(self.__target, self.__prediction,
                                              self.__parent_dir, self.__sub_dir)

    def save_results(self):
        """
        Saves test dataset with target values and prediction results with errors
        """
        self.__saving_strategy.save_results(self.__test_set, self.__target, self.__prediction,
                                            self.__parent_dir, self.__sub_dir)
