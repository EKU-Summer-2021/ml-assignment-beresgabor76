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
    def __init__(self, test_size=0.2, random_state=20):
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
        self.test_data = None

    def prepare(self):
        """
        Preparing train and test datasets for evaluation
        """
        try:
            self.__read_file()
        except:
            print('An error occurred during reading csv file!')
            sys.exit(1)
        self.__category_to_number()
        self.__split_dataset()
        self.__normalize_data()

    def __read_file(self):
        """
        Read csv file from ../data directory and splits to X and y
        """
        self.__dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data', 'insurance.csv'))
        self.__dataset_y = self.__dataset['charges']
        self.__dataset_x = self.__dataset.drop('charges', axis=1)

    def __category_to_number(self):
        """
        Converts categorical data types to numbers in X
        """
        self.__dataset_x.replace(to_replace='male', value=0, inplace=True)
        self.__dataset_x.replace(to_replace='female', value=1, inplace=True)
        self.__dataset_x.replace(to_replace='no', value=0, inplace=True)
        self.__dataset_x.replace(to_replace='yes', value=1, inplace=True)
        dataset_cat = self.__dataset_x[['region']]
        cat_encoder = OneHotEncoder()
        arr_cat_1hot = cat_encoder.fit_transform(dataset_cat)
        df_cat_1hot = pd.DataFrame(arr_cat_1hot.toarray(), columns=cat_encoder.get_feature_names())
        self.__dataset_x = pd.concat([self.__dataset_x, df_cat_1hot], axis=1).drop('region', axis=1)

    def __split_dataset(self):
        """
        Splits prepared datasets X,y to train and test datasets
        """
        dataset = pd.concat([self.__dataset_x, self.__dataset_y], axis=1)
        train_set, test_set = train_test_split(dataset,
                                               test_size=self.__test_size,
                                               random_state=self.__random_state)
        self.train_set_x = train_set.drop('charges', axis=1).copy()
        self.train_set_y = train_set['charges'].copy()
        self.test_set_x = test_set.drop('charges', axis=1).copy()
        self.test_set_y = test_set['charges'].copy()

    def __normalize_data(self):
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
        dataset = pd.concat([self.__dataset_x, self.__dataset_y], axis=1)
        corr_matrix = dataset.corr()
        print('Correlation values with charges attribute:')
        print(corr_matrix['charges'].sort_values(ascending=False))
