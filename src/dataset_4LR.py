"""
Module for dataset class used in Linear Regression
"""
from abc import ABC
import pandas as pd
from src import Dataset4SL
from sklearn.preprocessing import MinMaxScaler


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
        self.test_data = self.test_set_x.copy()
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        x_scaled_arr = scaler.fit_transform(self.train_set_x)
        self.train_set_x = pd.DataFrame(x_scaled_arr, columns=self.train_set_x.columns)
        x_scaled_arr = scaler.transform(self.test_set_x)
        self.test_set_x = pd.DataFrame(x_scaled_arr, columns=self.test_set_x.columns)

    def print_correlation(self):
        """
        Prints out correlation values between input and output data
        """
        dataset = pd.concat([self._dataset_x, self._dataset_y], axis=1)
        corr_matrix = dataset.corr()
        print('Correlation values with charges attribute:')
        print(corr_matrix['charges'].sort_values(ascending=False))