"""
Module for clustering algorithm DBSCAN
"""
import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from src.learning_algorithm import LearningAlgorithm
from src.decision_tree import DecisionTree
from src.saving_strategy_sl import SavingStrategy4SL
from src.plotting_strategy_clf import PlottingStrategy4CLF


class DbscanClustering(LearningAlgorithm):
    """
    Class for DBSCAN clustering algorithm
    """
    def __init__(self, eps, min_samples, saving_strategy, plotting_strategy):
        super().__init__(saving_strategy, plotting_strategy)
        self.__eps = eps
        self.__min_samples = min_samples
        self.__dbscan = None
        self._is_scaled_x = True
        self._parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '../results/clustering')
        self._sub_dir = self._make_save_dir()
        self._logger = self._setup_logger(f'DbscanLog{self._sub_dir}',
                                          os.path.join(self._parent_dir, self._sub_dir, 'run.log'))
        self._logger.info('Parameters: eps=%f, min_samples=%f', self.__eps, self.__min_samples)

    def clustering(self, dataset, scaler):
        """
        Creates clusters from the feature-scaled dataset, and put the labels into prediction
        """
        self._copy_datasets(dataset, dataset.index)
        self._x_scaler = scaler
        self.__dbscan = DBSCAN(eps=self.__eps, min_samples=self.__min_samples)
        self.__dbscan.fit(dataset)
        self._test_set_y = pd.DataFrame(self.__dbscan.labels_, columns=['Label'])
        self._prediction = pd.DataFrame(self.__dbscan.labels_, columns=['Label'])

    def test_clustering(self):
        """
        Tests the clustering with the use of a decision tree classifier
        """
        dataset = pd.concat([self._test_set_x, self._test_set_y], axis=1)
        dataset = dataset[dataset['Label'] != -1]
        train_set, test_set = train_test_split(dataset,
                                               test_size=0.2,
                                               random_state=30)
        train_set_x = train_set.drop(train_set.columns[-1], axis=1).copy()
        train_set_y = train_set[train_set.columns[-1]].copy()
        test_set_x = test_set.drop(test_set.columns[-1], axis=1).copy()
        test_set_y = test_set[test_set.columns[-1]].copy()
        tree_clf = DecisionTree(SavingStrategy4SL(), PlottingStrategy4CLF())
        tree_clf.train(train_set_x, train_set_y)
        tree_clf.test(test_set_x, test_set_y)
        tree_clf.save_results()
        tree_clf.plot_results()
