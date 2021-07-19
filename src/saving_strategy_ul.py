"""
Module for results' SavingStrategy for datasets of Unsupervised Learning
"""
import os
import pandas as pd
from src.saving_strategy import SavingStrategy


class SavingStrategy4UL(SavingStrategy):
    """
    Results' saving strategy implementation for Supervised Learning
    """
    def save_results(self, unscaled_test_set_x, test_set_y, prediction, save_dir):
        results_df = pd.concat([unscaled_test_set_x, prediction], axis=1)
        results_df = results_df.round(2)
        results_df.sort_values(by=['Label'], inplace=True)
        csv_file = os.path.join(os.path.dirname(__file__), save_dir, 'results.csv')
        results_df.to_csv(path_or_buf=csv_file, index=False)
