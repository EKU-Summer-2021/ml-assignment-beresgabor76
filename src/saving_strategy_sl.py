"""
Module for results' SavingStrategy for datasets of Supervised Learning
"""
import os
import pandas as pd
from src.saving_strategy import SavingStrategy


class SavingStrategy4SL(SavingStrategy):
    """
    Results' saving strategy implementation for Supervised Learning
    """
    def save_results(self, test_data, target, prediction, save_dir):
        """
        Saves results to given directory in csv format
        """
        results_df = pd.concat([test_data, target, prediction], axis=1)
        results_df['error'] = results_df.iloc[:, -1] - results_df.iloc[:, -2]
        results_df = results_df.round(2)
        csv_file = os.path.join(os.path.dirname(__file__), save_dir, 'results.csv')
        results_df.to_csv(path_or_buf=csv_file, index=False)

    def print_result(self, test_data, target, prediction):
        """
        Prints out results to console
        """
        results_df = pd.concat([test_data, target, prediction], axis=1)
        results_df['error'] = results_df.iloc[:, -1] - results_df.iloc[:, -2]
        results_df = results_df.round(2)
        print(results_df)
