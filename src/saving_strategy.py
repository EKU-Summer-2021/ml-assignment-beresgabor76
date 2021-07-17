"""
Module for results' SavingStrategy interface
"""


class SavingStrategy:
    """
    Results saving strategy informal interface
    """
    def save_results(self, test_data, target, prediction, parent_dir, sub_dir):
        pass
