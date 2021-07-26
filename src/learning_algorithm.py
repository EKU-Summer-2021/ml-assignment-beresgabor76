"""
Module for an abstract class for learning algorithm implementations
"""
from abc import ABC
import os
import logging
import datetime
import pandas as pd


class LearningAlgorithm(ABC):
    """
    Abstract parent class for the specific machine learning algorithms
    """
    def __init__(self, saving_strategy, plotting_strategy):
        super().__init__()
        self._test_set_x = None
        self._test_set_y = None
        self._is_scaled_x = False
        self._x_scaler = None
        self._is_scaled_y = False
        self._y_scaler = None
        self._prediction = None
        self._parent_dir = None
        self._sub_dir = None
        self.__saving_strategy = saving_strategy
        self.__plotting_strategy = plotting_strategy

    def _copy_datasets(self, test_set_x, test_set_y):
        self._test_set_x = pd.DataFrame(test_set_x.copy())
        self._test_set_x.reset_index(inplace=True)
        self._test_set_x.drop('index', axis=1, inplace=True)
        self._test_set_y = pd.DataFrame(test_set_y)
        self._test_set_y.reset_index(inplace=True)
        self._test_set_y.drop('index', axis=1, inplace=True)

    def _make_save_dir(self):
        """
        Makes a subdirectory for storing result and plot
        """
        resolution = datetime.timedelta(seconds=5)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        sub_dir = save_time.strftime('%Y.%m.%d_%H.%M.%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), self._parent_dir, sub_dir)):
            os.chdir(self._parent_dir)
            os.mkdir(sub_dir)
        return sub_dir

    def _setup_logger(self, name, log_file, level=logging.INFO):
        """
        Setup loggers for the learning algorithms
        """
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    def plot_results(self):
        """
        Plots out how the predicted values approximate the real ones
        """
        if self._is_scaled_y:
            unscaled_test_set_y = pd.DataFrame(self._y_scaler.inverse_transform(self._test_set_y),
                                               columns=self._test_set_y.columns)
            unscaled_prediction = pd.DataFrame(self._y_scaler.inverse_transform(self._prediction),
                                               columns=self._prediction.columns)
        else:
            unscaled_test_set_y = self._test_set_y
            unscaled_prediction = self._prediction
        self.__plotting_strategy.plot_results(unscaled_test_set_y, unscaled_prediction,
                                              self._parent_dir + '/' + self._sub_dir)

    def plot_clusters(self):
        """
        Plots out how the predicted values approximate the real ones
        """
        self.__plotting_strategy.plot_clusters(self._test_set_x, self._prediction,
                                               self._parent_dir + '/' + self._sub_dir)

    def save_results(self):
        """
        Saves test dataset with target values and prediction results with errors
        """
        if self._is_scaled_x:
            unscaled_test_set_x = pd.DataFrame(self._x_scaler.inverse_transform(self._test_set_x),
                                               columns=self._test_set_x.columns)
        else:
            unscaled_test_set_x = self._test_set_x
        if self._is_scaled_y:
            unscaled_test_set_y = pd.DataFrame(self._y_scaler.inverse_transform(self._test_set_y),
                                               columns=self._test_set_y.columns)
            unscaled_prediction = pd.DataFrame(self._y_scaler.inverse_transform(self._prediction),
                                               columns=self._prediction.columns)
        else:
            unscaled_test_set_y = self._test_set_y
            unscaled_prediction = self._prediction
        self.__saving_strategy.save_results(unscaled_test_set_x,
                                            unscaled_test_set_y, unscaled_prediction,
                                            self._parent_dir + '/' + self._sub_dir)
