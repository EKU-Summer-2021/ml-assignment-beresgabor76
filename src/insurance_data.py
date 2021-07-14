"""
Module for importing and preparing insurance data
"""
import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class InsuranceData:
    """
    Class for storing and preparing insurance data for evaluation
    """
    def __init__(self, test_size=0.2, random_state=12):
        """
        Constructor for creating member variables
        """
        self.__test_size = test_size
        self.__random_state = random_state
        self.__dataset = None
        self.__dataset_x = None
        self.__dataset_y = None
        self.train_set_x = None
        self.train_set_y = None
        self.test_set_x = None
        self.test_set_y = None

    def prepare(self):
        """
        Preparing train and test datasets for evaluation
        """
        try:
            self.__read_file()
        except:
            print('Error occurred during reading csv file!')
            sys.exit(1)
        self.__category_to_number()
        self.__normalize_data()
        self.__split_dataset()

    def __read_file(self):
        """
        Read csv file from ../data directory
        """
        self.__dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data', 'insurance.csv'))

    def __category_to_number(self):
        """
        Converts categorical data types to numbers and stores new dataset in member variables
        """
        self.__dataset_y = self.__dataset['charges']
        self.__dataset_x = self.__dataset.drop('charges', axis=1)
        self.__dataset_x.replace(to_replace='male', value=0, inplace=True)
        self.__dataset_x.replace(to_replace='female', value=1, inplace=True)
        self.__dataset_x.replace(to_replace='no', value=0, inplace=True)
        self.__dataset_x.replace(to_replace='yes', value=1, inplace=True)
        dataset_cat = self.__dataset_x[['region']]
        cat_encoder = OneHotEncoder()
        arr_cat_1hot = cat_encoder.fit_transform(dataset_cat)
        df_cat_1hot = pd.DataFrame(arr_cat_1hot.toarray(), columns=cat_encoder.get_feature_names())
        self.__dataset_x = pd.concat([self.__dataset_x, df_cat_1hot], axis=1).drop('region', axis=1)

    def __normalize_data(self):
        """
        Scales down all input data to [0, 1] interval
        """
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        x_scaled_arr = scaler.fit_transform(self.__dataset_x)
        self.__dataset_x = pd.DataFrame(x_scaled_arr, columns=self.__dataset_x.columns)

    def __split_dataset(self):
        """
        Splits prepared dataset to train and test datasets
        """
        dataset = pd.concat([self.__dataset_x, self.__dataset_y], axis=1)
        train_set, test_set = train_test_split(dataset, test_size=self.__test_size, random_state=self.__random_state)
        self.train_set_x = train_set.drop('charges', axis=1)
        self.train_set_y = train_set['charges']
        self.test_set_x = test_set.drop('charges', axis=1)
        self.test_set_y = test_set['charges']

    def print_correlation(self):
        """
        Prints out correlation values between input and output data
        """
        dataset = pd.concat([self.__dataset_x, self.__dataset_y], axis=1)
        corr_matrix = dataset.corr()
        print(corr_matrix['charges'].sort_values(ascending=False))
