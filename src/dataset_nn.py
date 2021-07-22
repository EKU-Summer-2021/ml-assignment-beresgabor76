"""
Module for dataset class used in Linear Regression
"""
from abc import ABC
import numpy as np
import pandas as pd
from src.dataset_sl import Dataset4SL


class Dataset4NN(Dataset4SL, ABC):
    """
    Class for dataset to Neural Network derived from Dataset4SL class
    """
    def __init__(self, filename, test_size, random_state):
        super().__init__(filename, test_size, random_state)

    def _feature_scaling(self):
        """
        Scales down all input data to [0, 1] interval, makes a copy of original data
        """
        self.x_scaler.fit(self.train_set_x)
        scaled_arr = self.x_scaler.transform(self.train_set_x)
        self.train_set_x = pd.DataFrame(scaled_arr, columns=self.train_set_x.columns)
        self.y_scaler.fit(np.array([self.train_set_y]).reshape(-1, 1))
        scaled_arr = self.y_scaler.transform(np.array([self.train_set_y]).reshape(-1, 1))
        self.train_set_y = np.ravel(scaled_arr)
        scaled_arr = self.x_scaler.transform(self.test_set_x)
        self.test_set_x = pd.DataFrame(scaled_arr, columns=self.test_set_x.columns)
        scaled_arr = self.y_scaler.transform(np.array([self.test_set_y]).reshape(-1, 1))
        self.test_set_y = pd.DataFrame(scaled_arr)

