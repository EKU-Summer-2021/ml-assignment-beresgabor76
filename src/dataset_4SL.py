"""
Module for class implemented for reading and preparing dataset for supervised learning
"""
import sys
import os
from abc import ABC
from abc import abstractmethod
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


class Dataset4SL(ABC):
    """
    Abstract class implementing template pattern for storing and preparing data
    for supervised learning
    """
    def __init__(self, filename, test_size=0.2, random_state=30):
        super().__init__()
        self.__filename = filename
        self.__test_size = test_size
        self.__random_state = random_state
        self._dataset_x = None
        self._dataset_y = None
        self.train_set_x = None
        self.train_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        self.test_data = None

    def prepare(self):
        self.__read_file()
        self._categories_encoding()
        self.__split_dataset()
        self._feature_scaling()

    def __read_file(self):
        try:
            dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data', self.__filename))
            self._dataset_x = dataset.iloc[:, :-1]
            self._dataset_y = dataset.iloc[:, -1:]
        except:
            print('An error occurred during reading the csv file.')
            sys.exit(1)

    @abstractmethod
    def _categories_encoding(self):
        pass

    def _category_ordinal_encoder(self, column_name):
        dataset_cat = self._dataset_x[[column_name]]
        cat_encoder = OrdinalEncoder()
        arr_cat_ordinal = cat_encoder.fit_transform(dataset_cat)
        df_cat_ordinal = pd.DataFrame(arr_cat_ordinal, columns=[column_name])
        self._dataset_x = self._dataset_x.drop(column_name, axis=1)
        self._dataset_x = pd.concat([self._dataset_x, df_cat_ordinal], axis=1)

    def _category_1hot_encoder(self, column_name):
        dataset_cat = self._dataset_x[[column_name]]
        cat_encoder = OneHotEncoder()
        arr_cat_1hot = cat_encoder.fit_transform(dataset_cat)
        df_cat_1hot = pd.DataFrame(arr_cat_1hot.toarray(), columns=cat_encoder.get_feature_names())
        self._dataset_x = pd.concat([self._dataset_x, df_cat_1hot], axis=1)\
            .drop(column_name, axis=1)

    def __split_dataset(self, test_size=0.2, random_state=None):
        self.__test_size = test_size
        if random_state is not None:
            self.__random_state = random_state
        dataset = pd.concat([self._dataset_x, self._dataset_y], axis=1)
        train_set, test_set = train_test_split(dataset,
                                               test_size=self.__test_size,
                                               random_state=self.__random_state)
        self.train_set_x = train_set.drop(train_set.columns[-1], axis=1).copy()
        self.train_set_y = train_set[train_set.columns[-1]].copy()
        self.test_set_x = test_set.drop(test_set.columns[-1], axis=1).copy()
        self.test_set_y = test_set[test_set.columns[-1]].copy()

    @abstractmethod
    def _feature_scaling(self):
        pass