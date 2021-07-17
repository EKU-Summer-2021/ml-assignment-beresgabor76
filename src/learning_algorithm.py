"""
Module for an abstract class for learning algorithm implementations
"""
from abc import ABC
import os
import datetime


class LearningAlgorithm(ABC):
    """
    Abstract parent class for the specific machine learning algorithms
    """
    def __init__(self, saving_strategy, plotting_strategy):
        super().__init__()
        self._test_set = None
        self._target = None
        self._prediction = None
        self._parent_dir = None
        self._sub_dir = None
        self.__saving_strategy = saving_strategy
        self.__plotting_strategy = plotting_strategy

    def _make_save_dir(self):
        """
        Makes a subdirectory for storing result and plot
        """
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        sub_dir = save_time.strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), self._parent_dir, sub_dir)):
            os.chdir(self._parent_dir)
            os.mkdir(sub_dir)
        return sub_dir

    def plot_results(self):
        """
        Plots out how the predicted values approximate the real ones
        """
        self.__plotting_strategy.plot_results(self._target, self._prediction,
                                              self._parent_dir + '/' + self._sub_dir)

    def save_results(self):
        """
        Saves test dataset with target values and prediction results with errors
        """
        self.__saving_strategy.save_results(self._test_set, self._target, self._prediction,
                                            self._parent_dir + '/' + self._sub_dir)
