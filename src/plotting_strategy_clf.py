"""
Module for results' PlottingStrategy for datasets of Classification
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from src.plotting_strategy import PlottingStrategy


class PlottingStrategy4CLF(PlottingStrategy):
    """
    Results' plotting strategy implementation for Classification
    """
    def plot_results(self, test_set_y, prediction, save_dir):
        """
        Plots out a histogram as well as the confusion matrix fpr the test
        """
        fig = plt.figure()
        plt.hist(pd.concat([test_set_y, prediction], axis=1),
                 color=['blue', 'red'], label=['Actual', 'Prediction'], histtype='bar')
        plt.legend()
        plot_file = os.path.join(os.path.dirname(__file__), save_dir, 'histogram.png')
        fig.savefig(plot_file)
        plot_file = os.path.join(os.path.dirname(__file__), save_dir, 'confusion_mx.png')
        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(test_set_y, prediction),
                               display_labels=np.sort(test_set_y.iloc[:, 0].unique())).plot()
        plt.savefig(plot_file)

    def plot_show(self, test_set_y, prediction):
        """
        Shows plot results of learning algorithm's test
        """
