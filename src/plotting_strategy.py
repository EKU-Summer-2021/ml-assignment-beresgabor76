"""
Module for results' PlottingStrategy interface
"""


class PlottingStrategy:
    """
    Results plotting strategy informal interface
    """
    def plot_results(self, target, prediction, save_dir):
        """
        Plots results of learning algorithm's test
        """
        pass
