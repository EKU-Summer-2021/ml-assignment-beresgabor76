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
    def plot_results(self, target, prediction, parent_dir, sub_dir):
        """
        Plots out how the predicted values approximate the real ones
        """
        fig = plt.figure()
        min_x = min_y = target.min()
        max_x = max_y = target.max()
        plt.plot([min_x, max_x], [min_y, max_y])
        plt.scatter(target, prediction, alpha=0.5)
        plot_file = os.path.join(os.path.dirname(__file__), parent_dir, sub_dir, 'results.png')
        fig.savefig(plot_file)
