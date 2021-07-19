"""
Module for Dataset class for Students' performance
"""
from src import Dataset4UL


class Dataset4ULStudentsPerformance(Dataset4UL):
    """
    Class for storing and preparing Students' Performance data for Clustering
    """
    def __init__(self):
        super().__init__('StudentsPerformance.csv')

    def _categories_encoding(self):
        self._category_ordinal_encoder('gender')
        self._category_1hot_encoder('race/ethnicity')
        self._category_ordinal_encoder('parental level of education',
                                       [["some high school",
                                         "high school",
                                         "some college",
                                         "associate's degree",
                                         "bachelor's degree",
                                         "master's degree"]])
        self._category_ordinal_encoder('lunch',
                                       [['free/reduced', 'standard']])
        self._category_ordinal_encoder('test preparation course',
                                       [['none', 'completed']])

