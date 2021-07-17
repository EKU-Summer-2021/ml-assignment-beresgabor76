"""
Module for results' PlottingStrategy interface
"""


class PlottingStrategy:
    """
    Results plotting strategy informal interface
    """
    def plot_results(self, target, prediction, parent_dir, sub_dir):
        pass
