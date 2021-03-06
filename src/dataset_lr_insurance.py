"""
Module for class intended to store and prepare insurance data for linear regression
"""
from src.dataset_lr import Dataset4LR


class Dataset4LRInsurance(Dataset4LR):
    """
    Class intended to store and prepare insurance data for linear regression
    """

    def __init__(self, test_size=0.2, random_state=30):
        super().__init__('insurance.csv', test_size, random_state)

    def _categories_encoding(self):
        self._category_ordinal_encoder('sex')
        self._category_ordinal_encoder('smoker')
        self._category_1hot_encoder('region')
