"""
Module for results' PlottingStrategy interface
"""


class PlottingStrategy:
    """
    Results plotting strategy informal interface
    """
    def plot_results(self, test_set_y, prediction, save_dir):
        """
        Saves plot results of learning algorithm's test to png file
        """

    def plot_show(self, test_set_y, prediction):
        """
        Shows plot results of learning algorithm's test
        """

    def plot_clusters(self, test_set_x, prediction, save_dir):
        """
        Plots out the clusters as scatter plot to a png file
        """