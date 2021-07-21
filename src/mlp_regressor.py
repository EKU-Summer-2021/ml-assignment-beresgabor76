"""
Module for MLPRegressor wrapping class
"""
import os
import pandas as pd
from src.learning_algorithm import LearningAlgorithm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


class MlpRegressor(LearningAlgorithm):
    """
    Class for wrapping MLPRegressor
    """
    def __init__(self, saving_strategy, plotting_strategy):
        super().__init__(saving_strategy, plotting_strategy)
        self._parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '../results/mlp_regression')
        self._sub_dir = self._make_save_dir()
        self._logger = self._setup_logger(f'MlpRegressionLog{self._sub_dir}',
                                          os.path.join(self._parent_dir, self._sub_dir, 'run.log'))
        self.__activation = 'relu'
        self.__hidden_layer_sizes = (25, 50, 100, 50, 25)
        self.__learning_rate = 'adaptive'
        self.__learning_rate_init = 0.001
        self.__momentum = 0.9
        self.__max_iter = 20000
        self.__mlp = MLPRegressor(hidden_layer_sizes=self.__hidden_layer_sizes,
                                  activation=self.__activation,
                                  solver='adam',
                                  alpha=0.0001,
                                  learning_rate=self.__learning_rate,
                                  learning_rate_init=self.__learning_rate_init,
                                  power_t=0.5,
                                  max_iter=self.__max_iter,
                                  shuffle=False,
                                  random_state=None,
                                  tol=0.0001,
                                  verbose=False,
                                  warm_start=False,
                                  momentum=self.__momentum,
                                  nesterovs_momentum=True,
                                  early_stopping=False,
                                  validation_fraction=0.2,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-08,
                                  n_iter_no_change=10,
                                  max_fun=50000)

    def __log_params(self):
        params = {'activation': self.__activation,
                  'hidden_layer_sizes': self.__hidden_layer_sizes,
                  'learning_rate': self.__learning_rate,
                  'learning_rate_init': self.__learning_rate_init,
                  'momentum': self.__momentum,
                  'max_iter': self.__max_iter}
        self._logger.info('Set parameters for MLPRegressor:')
        self._logger.info(params)
        self._logger.info('All parameters for MLPRegressor:')
        self._logger.info(self.__mlp.get_params())

    def determine_parameters(self, train_set_x, train_set_y):
        """
        GridSearchCV
        """
        params = {'activation': ['relu', 'identity'],
                  'hidden_layer_sizes': [(5, 10, 20, 10, 5), (20, 50, 100, 50, 20)],
                  'learning_rate_init': [0.001, 0.01],
                  'momentum': [0.9, 0.5]}
        grid_search = GridSearchCV(estimator=self.__mlp,
                                   param_grid=params,
                                   scoring='neg_mean_squared_error',
                                   refit=True)
        grid_search.fit(train_set_x, train_set_y)
        self._logger.info('Best parameters found by GridSearchCV:')
        self._logger.info(grid_search.best_params_)
        best_params = {'activation': grid_search.best_params_['activation'],
                       'hidden_layer_sizes': grid_search.best_params_['hidden_layer_sizes'],
                       'learning_rate_init': grid_search.best_params_['learning_rate_init'],
                       'momentum': grid_search.best_params_['momentum']}
        self.__mlp.set_params(**best_params)

    def train(self, train_set_x, train_set_y):
        """
        Trains the neural network
        """
        self.__log_params()
        self.__mlp.fit(train_set_x, train_set_y)
        self._logger.info('Iterations during training ann: %d', self.__mlp.n_iter_)

    def test(self, unscaled_test_set_x, test_set_x, test_set_y):
        """
        Tests the neural network
        """
        self._copy_datasets(unscaled_test_set_x, test_set_x, test_set_y)
        self._prediction = pd.DataFrame(self.__mlp.predict(test_set_x),
                                        index=test_set_x.index,
                                        columns=['prediction'])
        self._prediction.reset_index(inplace=True)
        self._prediction = self._prediction.drop('index', axis=1)
        self._logger.info('Score for test prediction: %f',
                          self.__mlp.score(test_set_x, test_set_y))
