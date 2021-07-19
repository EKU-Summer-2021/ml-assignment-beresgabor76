"""
Module for results' SavingStrategy interface
"""


class SavingStrategy:
    """
    Results saving strategy informal interface
    """
    def save_results(self, test_data, target, prediction, save_dir):
        """
        Saves results to csv file
        """

    def print_result(self, test_data, target, prediction):
        """
        Prints out results to console
        """
