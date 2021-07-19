"""
Module for class implemented for reading and preparing dataset for unsupervised learning
"""
import sys
import os
from abc import ABC
from abc import abstractmethod
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler


class Dataset4UL(ABC):
    """
    Abstract class implementing template pattern for storing and preparing data
    for unsupervised learning
    """
    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self.unscaled_dataset = None
        self.dataset = None

    def prepare(self):
        """
        Prepares dataset for machine learning process
        """
        self.__read_file()
        self._categories_encoding()
        self._feature_scaling()

    def __read_file(self):
        """
        Reads in a csv file and splits into X and y sets
        """
        try:
            df_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data', self._filename))
            self.unscaled_dataset = df_dataset.copy()
            self.dataset = df_dataset.copy()
        except:
            print('An error occurred during reading the csv file.')
            sys.exit(1)

    @abstractmethod
    def _categories_encoding(self):
        """
        Used in child classes to encode category attributes
        """

    def _category_ordinal_encoder(self, column_name, categories='auto'):
        """
        Encodes a category attribute to ordinal numbers
        """
        dataset_cat = self.dataset[[column_name]]
        cat_encoder = OrdinalEncoder(categories=categories)
        arr_cat_ordinal = cat_encoder.fit_transform(dataset_cat)
        df_cat_ordinal = pd.DataFrame(arr_cat_ordinal, columns=[column_name])
        self.dataset.drop(column_name, axis=1, inplace=True)
        self.dataset = pd.concat([self.dataset, df_cat_ordinal], axis=1)

    def _category_1hot_encoder(self, column_name):
        """
        Encodes a category attribute to a set of attributes, one for each category value
        """
        dataset_cat = self.dataset[[column_name]]
        cat_encoder = OneHotEncoder()
        arr_cat_1hot = cat_encoder.fit_transform(dataset_cat)
        df_cat_1hot = pd.DataFrame(arr_cat_1hot.toarray(), columns=cat_encoder.get_feature_names())
        self.dataset = pd.concat([self.dataset, df_cat_1hot], axis=1)\
            .drop(column_name, axis=1)

    def _feature_scaling(self):
        """
        Scales down all input data to [0, 1] interval, makes a copy of original data
        """
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        x_scaled_arr = scaler.fit_transform(self.dataset)
        self.dataset = pd.DataFrame(x_scaled_arr, columns=self.dataset.columns)
