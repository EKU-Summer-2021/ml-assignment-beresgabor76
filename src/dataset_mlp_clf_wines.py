"""
Module for class intended to store and prepare insurance data for linear regression
"""
from src.dataset_lr import Dataset4LR


class Dataset4MlpClfWines(Dataset4LR):
    """
    Class intended to store and prepare insurance data for linear regression
    """

    def __init__(self, test_size=0.2, random_state=30):
        super().__init__('winequality-red.csv', test_size, random_state)

    def _categories_encoding(self):
        """
        No category attributes
        """