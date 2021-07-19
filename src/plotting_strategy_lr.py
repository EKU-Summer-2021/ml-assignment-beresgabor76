"""
Module for results' PlottingStrategy for datasets of Linear Regresssion
"""
import os
import matplotlib.pyplot as plt
from src.plotting_strategy import PlottingStrategy


class PlottingStrategy4LR(PlottingStrategy):
    """
    Results' plotting strategy implementation for Linear Regression
    """
    def plot_results(self, test_set_y, prediction, save_dir):
        """
        Plots out how the predicted values approximate the real ones
        """
        fig = plt.figure()
        min_x = min_y = test_set_y.min()
        max_x = max_y = test_set_y.max()
        plt.plot([min_x, max_x], [min_y, max_y])
        plt.scatter(test_set_y, prediction, alpha=0.5)
        plot_file = os.path.join(os.path.dirname(__file__), save_dir, 'results.png')
        fig.savefig(plot_file)

    def plot_show(self, test_set_y, prediction):
        """
        Shows plot results of learning algorithm's test
        """
        min_x = min_y = test_set_y.min()
        max_x = max_y = test_set_y.max()
        plt.plot([min_x, max_x], [min_y, max_y])
        plt.scatter(test_set_y, prediction, alpha=0.5)
        plt.show()
