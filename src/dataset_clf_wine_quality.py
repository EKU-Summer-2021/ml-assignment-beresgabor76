"""
Module for class intended to store and prepare wine quality data
for classification with Decision Tree
"""
from src.dataset_clf import Dataset4CLF


class Dataset4CLFWineQuality(Dataset4CLF):
    """
    Class intended to store and prepare wine quality data for classification with decision tree
    """

    def __init__(self, test_size=0.2, random_state=20):
        super().__init__('winequality-red.csv', test_size, random_state)

    def __repr__(self):
        return self.test_data

    def _categories_encoding(self):
        """
        No category attributes to be encoded
        """
