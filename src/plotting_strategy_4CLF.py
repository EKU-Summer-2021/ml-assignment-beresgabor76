"""
Module for results' PlottingStrategy for datasets of Classification
"""
from src.plotting_strategy import PlottingStrategy
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt


class PlottingStrategy4CLF(PlottingStrategy):
    """
    Results' plotting strategy implementation for Classification
    """

    def plot_results(self, target, prediction, parent_dir):
        """
        Plots out the tree as well as confusion matrix fpr the test
        """
        fig = plt.figure()
        plt.hist(pd.concat([target, prediction], axis=1),
                 color=['blue', 'red'], label=['Actual', 'Prediction'],
                 histtype='bar', bins=6)
        plt.legend()
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        sub_dir = save_time.strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), parent_dir, sub_dir)):
            os.chdir(parent_dir)
            os.mkdir(sub_dir)
        plot_file = os.path.join(os.path.dirname(__file__), parent_dir,
                                 sub_dir, 'results.png')
        fig.savefig(plot_file)
