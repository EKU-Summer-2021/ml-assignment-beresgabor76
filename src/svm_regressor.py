"""
Module for SVM Regression class
"""
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from src.learning_algorithm import LearningAlgorithm


class SvmRegressor(LearningAlgorithm):
    """
    Class for training an SVM Regression model, and then test it
    """

    def __init__(self, saving_strategy, plotting_strategy):
        """
        Constructor creates a SVM Regression model
        """
        super().__init__(saving_strategy, plotting_strategy)
        self.__svm_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
        self._is_scaled_x = True
        self._parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '../results/svm_regression')
        self._sub_dir = self._make_save_dir()
        self._logger = self._setup_logger(f'SvmRegressionLog{self._sub_dir}',
                                          os.path.join(self._parent_dir, self._sub_dir, 'run.log'))

    def determine_parameters(self, train_set_x, train_set_y):
        """
        GridSearchCV
        """
        params = {'kernel': ['poly', 'rbf'],
                  'degree': [2, 3, 4],
                  'C': [5, 3, 2],
                  'epsilon': [0.2, 0.1, 0.08]}
        grid_search = GridSearchCV(estimator=self.__svm_reg,
                                   param_grid=params,
                                   scoring='neg_mean_squared_error',
                                   refit=True)
        grid_search.fit(train_set_x, train_set_y)
        self._logger.info('Best parameters found by GridSearchCV:')
        self._logger.info(grid_search.best_params_)
        best_params = {'kernel': grid_search.best_params_['kernel'],
                       'degree': grid_search.best_params_['degree'],
                       'C': grid_search.best_params_['C'],
                       'epsilon': grid_search.best_params_['epsilon']}
        self.__svm_reg.set_params(**best_params)

    def set_parameters(self, kernel, degree=3, C=1, epsilon=0.1):
        """
        Sets parameters of SVM Regressor
        """
        params = {'kernel': kernel, 'degree': degree, 'C': C, 'epsilon': epsilon}
        self.__svm_reg.set_params(**params)

    def __log_parameters(self):
        """
        Logs set parameters to run.log
        """
        self._logger.info('\nSet parameters: %s', str(self.__svm_reg.get_params()))

    def train(self, train_set_x, train_set_y):
        """
        Trains the SVM Regression model with passed data
        """
        self.__svm_reg.fit(train_set_x, train_set_y)
        self.__log_parameters()

    def test(self, test_set_x, test_set_y, x_scaler, y_scaler):
        """
        Tests the SVM Regression model with passed data
        """
        self._copy_datasets(test_set_x, test_set_y)
        self._x_scaler = x_scaler
        self._y_scaler = y_scaler
        score = self.__svm_reg.score(self._test_set_x, self._test_set_y)
        self._logger.info('\nScore for test set: %f', score)
        self._prediction = pd.DataFrame(self.__svm_reg.predict(self._test_set_x),
                                        index=test_set_x.index,
                                        columns=['prediction'])
        self._prediction.reset_index(inplace=True)
        self._prediction = self._prediction.drop('index', axis=1)
