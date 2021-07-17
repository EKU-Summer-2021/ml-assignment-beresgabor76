"""
Module for results' PlottingStrategy for datasets of Classification
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.plotting_strategy import PlottingStrategy


class PlottingStrategy4CLF(PlottingStrategy):
    """
    Results' plotting strategy implementation for Classification
    """

    def plot_results(self, target, prediction, parent_dir, sub_dir):
        """
        Plots out the tree as well as confusion matrix fpr the test
        """
        fig = plt.figure()
        plt.hist(pd.concat([target, prediction], axis=1),
                 color=['blue', 'red'], label=['Actual', 'Prediction'],
                 histtype='bar', bins=6)
        plt.legend()
        plot_file = os.path.join(os.path.dirname(__file__), parent_dir,
                                 sub_dir, 'results.png')
        fig.savefig(plot_file)
