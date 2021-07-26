"""
Module for Linear Regression class
"""
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.learning_algorithm import LearningAlgorithm


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
        self._is_scaled_x = True
        self._parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '../results/linear_regression')
        self._sub_dir = self._make_save_dir()
        self._logger = self._setup_logger(f'LinearRegressionLog{self._sub_dir}',
                                          os.path.join(self._parent_dir, self._sub_dir, 'run.log'))

    def __correlation(self, dataset_x, dataset_y):
        """
        Returns correlation values between input and output data
        """
        dataset = pd.concat([dataset_x, dataset_y], axis=1)
        corr_matrix = dataset.corr()
        return corr_matrix['charges'].sort_values(ascending=False)

    def train(self, train_set_x, train_set_y):
        """
        Trains the LinearRegression model with passed data
        """
        self.__lin_reg.fit(train_set_x, train_set_y)
        self._logger.info('Correlation values with charges attribute in train set:')
        self._logger.info('\n%s', self.__correlation(train_set_x, train_set_y))

    def test(self, test_set_x, test_set_y, scaler):
        """
        Tests the LinearRegression model with passed data
        """
        self._copy_datasets(test_set_x, test_set_y)
        self._x_scaler = scaler
        score = self.__lin_reg.score(test_set_x, test_set_y)
        self._logger.info('\nScore for test set: %f', score)
        self._prediction = pd.DataFrame(self.__lin_reg.predict(test_set_x),
                                        index=test_set_x.index,
                                        columns=['prediction'])
        self._prediction.reset_index(inplace=True)
        self._prediction = self._prediction.drop('index', axis=1)
