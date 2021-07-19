"""
Module for results' PlottingStrategy interface
"""


class PlottingStrategy:
    """
    Results plotting strategy informal interface
    """
    def plot_results(self, target, prediction, save_dir):
        """
        Saves plot results of learning algorithm's test to png file
        """

    def plot_show(self, target, prediction):
        """
        Shows plot results of learning algorithm's test
        """
