'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from src.dataset_sl import Dataset4SL
from src.dataset_lr import Dataset4LR
from src.dataset_clf import Dataset4CLF
from src.dataset_lr_insurance import Dataset4LRInsurance
from src.dataset_clf_wine_quality import Dataset4CLFWineQuality
from src.dataset_ul import Dataset4UL
from src.dataset_ul_students import Dataset4ULStudentsPerformance
from src.saving_strategy import SavingStrategy
from src.saving_strategy_sl import SavingStrategy4SL
from src.saving_strategy_ul import SavingStrategy4UL
from src.plotting_strategy import PlottingStrategy
from src.plotting_strategy_lr import PlottingStrategy4LR
from src.plotting_strategy_clf import PlottingStrategy4CLF
from src.plotting_strategy_clu import PlottingStrategy4CLU
from src.learning_algorithm import LearningAlgorithm
from src.linear_regressor import LinearRegressor
from src.decision_tree import DecisionTree
from src.dbscan import DbscanClustering


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
