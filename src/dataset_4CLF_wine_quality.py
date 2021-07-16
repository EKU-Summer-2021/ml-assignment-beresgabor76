"""
Module for class intended to store and prepare wine quality data
for classification with Decision Tree
"""
from abc import ABC
from src import Dataset4CLF


class Dataset4CLFWineQuality(Dataset4CLF, ABC):
    """
    Class intended to store and prepare wine quality data for classification with decision tree
    """

    def __init__(self, test_size, random_state):
        super().__init__('winequality-red.csv', test_size, random_state)

    def _categories_encoding(self):
        """
        No category attributes to be encoded
        """
        pass