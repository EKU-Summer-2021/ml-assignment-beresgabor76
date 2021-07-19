"""
Module for results' SavingStrategy interface
"""


class SavingStrategy:
    """
    Results saving strategy informal interface
    """
    def save_results(self, unscaled_test_set_x, test_set_y, prediction, save_dir):
        """
        Saves results to csv file
        """

    def print_result(self, unscaled_test_set_x, test_set_y, prediction):
        """
        Prints out results to console
        """
