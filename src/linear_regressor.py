"""
Module for Linear Regression class
"""
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from src import LearningAlgorithm


class LinearRegressor(LearningAlgorithm):
    """
    Class for training a Linear Regression model, and then test it
    """

    def __init__(self, saving_strategy, plotting_strategy):
        """
        Constructor creates a LinearRegression model
        """
        super().__init__(saving_strategy, plotting_strategy)
        self.__lin_reg = LinearRegression()
        self._parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '../results/linear_regression')
        self._sub_dir = self._make_save_dir()

    def train(self, train_set_x, train_set_y):
        """
        Trains the LinearRegression model with passed data
        """
        self.__lin_reg.fit(train_set_x, train_set_y)

    def test(self, test_data, test_set_x, test_set_y):
        """
        Tests the LinearRegression model with passed data
        """
        self._test_data = test_data.copy()
        self._test_data.reset_index(inplace=True)
        self._test_data = self._test_data.drop('index', axis=1)
        self._test_set = pd.DataFrame(test_set_x.copy())
        self._test_set.reset_index(inplace=True)
        self._test_set = self._test_set.drop('index', axis=1)
        self._target = pd.DataFrame(test_set_y)
        self._target.reset_index(inplace=True)
        self._target = self._target.drop('index', axis=1)
        score = self.__lin_reg.score(test_set_x, test_set_y)
        print("\nScore for test set: " + str(score))
        self._prediction = pd.DataFrame(self.__lin_reg.predict(test_set_x),
                                         index=test_set_x.index,
                                         columns=['prediction'])
        self._prediction.reset_index(inplace=True)
        self._prediction = self._prediction.drop('index', axis=1)

