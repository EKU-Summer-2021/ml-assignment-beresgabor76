"""
Module for dataset class used in Classification task
"""
from abc import ABC

from src.dataset_sl import Dataset4SL


class Dataset4CLF(Dataset4SL, ABC):
    """
    Class for Classification task derived from Dataset4SL class
    """
    def __init__(self, filename, test_size, random_state):
        super().__init__(filename, test_size, random_state)

    def _feature_scaling(self):
        """
        No need for feature scaling before classification
        """
        self.test_data = self.test_set_x.copy()
