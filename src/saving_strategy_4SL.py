"""
Module for results' SavingStrategy for datasets of Supervised Learning
"""
import os
import datetime
import pandas as pd
from src.saving_strategy import SavingStrategy


class SavingStrategy4SL(SavingStrategy):
    """
    Results' saving strategy implementation for Supervised Learning
    """
    def save_results(self, test_data, target, prediction, parent_dir):
        results_df = pd.concat([test_data, target, prediction], axis=1)
        results_df['error'] = results_df.iloc[:, -1] - results_df.iloc[:, -2]
        results_df = results_df.round(2)
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        sub_dir = save_time.strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), parent_dir, sub_dir)):
            os.chdir(parent_dir)
            os.mkdir(sub_dir)
        csv_file = os.path.join(os.path.dirname(__file__), parent_dir, sub_dir, 'results.csv')
        results_df.to_csv(path_or_buf=csv_file, index=False)
