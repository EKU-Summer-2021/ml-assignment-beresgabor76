'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from dataset_sl import Dataset4SL
from dataset_lr import Dataset4LR
from dataset_clf import Dataset4CLF
from dataset_lr_insurance import Dataset4LRInsurance
from dataset_clf_wine_quality import Dataset4CLFWineQuality
from dataset_ul import Dataset4UL
from dataset_ul_students import Dataset4ULStudentsPerformance
from saving_strategy import SavingStrategy
from saving_strategy_sl import SavingStrategy4SL
from saving_strategy_ul import SavingStrategy4UL
from plotting_strategy import PlottingStrategy
from plotting_strategy_lr import PlottingStrategy4LR
from plotting_strategy_clf import PlottingStrategy4CLF
from plotting_strategy_clu import PlottingStrategy4CLU
from learning_algorithm import LearningAlgorithm
from linear_regressor import LinearRegressor
from decision_tree import DecisionTree
from dbscan import DbscanClustering


__all__ = [
    'Dataset4SL',
    'Dataset4LR',
    'Dataset4CLF',
    'Dataset4LRInsurance',
    'Dataset4CLFWineQuality',
    'Dataset4UL',
    'Dataset4ULStudentsPerformance',
    'SavingStrategy',
    'SavingStrategy4SL',
    'SavingStrategy4UL',
    'PlottingStrategy',
    'PlottingStrategy4LR',
    'PlottingStrategy4CLF',
    'PlottingStrategy4CLU',
    'LearningAlgorithm',
    'LinearRegressor',
    'DecisionTree',
    'DbscanClustering'
]
