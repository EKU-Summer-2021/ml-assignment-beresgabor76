"""
Module for MLPClassifier wrapping class
"""
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from src.learning_algorithm import LearningAlgorithm


class MlpNetwork(LearningAlgorithm):
    """
    Class for wrapping MLPRegressor or MLPClassifier
    """
    def __init__(self, neural_network, saving_strategy, plotting_strategy):
        super().__init__(saving_strategy, plotting_strategy)
        self.__mlp = neural_network
        self.__activation = 'relu'
        self.__hidden_layer_sizes = (5, 10, 20, 10, 5)
        self.__learning_rate = 'adaptive'
        self.__max_iter = 5000
        self.set_parameters(self.__activation, self.__hidden_layer_sizes,
                            self.__learning_rate, self.__max_iter)
        self._is_scaled_x = True
        if isinstance(self.__mlp, MLPClassifier):
            self._parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '../results/mlp_classifier')
            self._scoring = 'roc_auc_ovr'
        elif isinstance(self.__mlp, MLPRegressor):
            self._parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '../results/mlp_regression')
            self._scoring = 'neg_mean_squared_error'
        self._sub_dir = self._make_save_dir()
        self._logger = self._setup_logger(f'MlpNetworkLog.{self._parent_dir}.{self._sub_dir}',
                                          os.path.join(self._parent_dir, self._sub_dir, 'run.log'))

    def set_parameters(self, activation, hidden_layer_sizes, learning_rate, max_iter):
        """
        Sets parameters of MLPRegressor or MLPClassifier instance
        """
        self.__activation = activation
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter
        params = {'activation': self.__activation,
                  'hidden_layer_sizes': self.__hidden_layer_sizes,
                  'learning_rate': self.__learning_rate,
                  'max_iter': self.__max_iter}
        self.__mlp.set_params(**params)

    def __log_params(self):
        """
        Logs MLPRegressor or MLPClassifier training parameters
        """
        self._logger.info('Set parameters for MLP Network:')
        self._logger.info(self.__mlp.get_params())

    def determine_parameters(self, train_set_x, train_set_y):
        """
        GridSearchCV
        """
        params = {'activation': ['relu', 'identity'],
                  'hidden_layer_sizes': [(5, 10, 5), (5, 10, 20, 10, 5), (25, 50, 100, 50, 25)]}
        grid_search = GridSearchCV(estimator=self.__mlp,
                                   param_grid=params,
                                   scoring=self._scoring,
                                   refit=True)
        grid_search.fit(train_set_x, train_set_y)
        self._logger.info('Best parameters found by GridSearchCV:')
        self._logger.info(grid_search.best_params_)
        self.__activation = grid_search.best_params_['activation']
        self.__hidden_layer_sizes = grid_search.best_params_['hidden_layer_sizes']
        best_params = {'activation': self.__activation,
                       'hidden_layer_sizes': self.__hidden_layer_sizes}
        self.__mlp.set_params(**best_params)

    def train(self, train_set_x, train_set_y):
        """
        Trains the neural network
        """
        self.__log_params()
        self.__mlp.fit(train_set_x, train_set_y)
        self._logger.info('Iterations during training ann: %d', self.__mlp.n_iter_)

    def test(self, test_set_x, test_set_y, x_scaler, y_scaler=None):
        """
        Tests the neural network
        """
        self._copy_datasets(test_set_x, test_set_y)
        self._x_scaler = x_scaler
        if y_scaler is not None:
            self._is_scaled_y = True
            self._y_scaler = y_scaler
        self._prediction = pd.DataFrame(self.__mlp.predict(test_set_x),
                                        index=test_set_x.index,
                                        columns=['prediction'])
        self._prediction.reset_index(inplace=True)
        self._prediction = self._prediction.drop('index', axis=1)
        self._logger.info('Score for test prediction: %f',
                          self.__mlp.score(self._test_set_x, self._test_set_y))
