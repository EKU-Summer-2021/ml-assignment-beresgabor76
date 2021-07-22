"""
Module for dataset class used in Linear Regression
"""
from abc import ABC
import pandas as pd
from src.dataset_sl import Dataset4SL


class Dataset4LR(Dataset4SL, ABC):
    """
    Class fpr Linear Regression derived from Dataset4SL class
    """
    def __init__(self, filename, test_size, random_state):
        super().__init__(filename, test_size, random_state)

    def _feature_scaling(self):
        """
        Scales down all input data to [0, 1] interval, makes a copy of original data
        """
        self.unscaled_test_set_x = self.test_set_x.copy()
        self.x_scaler.fit(self.train_set_x)
        x_scaled_arr = self.x_scaler.transform(self.train_set_x)
        self.train_set_x = pd.DataFrame(x_scaled_arr, columns=self.train_set_x.columns)
        x_scaled_arr = self.x_scaler.transform(self.test_set_x)
        self.test_set_x = pd.DataFrame(x_scaled_arr, columns=self.test_set_x.columns)
