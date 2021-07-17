'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from src.polynomial import Polynomial
from src.linear_regressor import LinearRegressor
from src.dataset_sl import Dataset4SL
from src.dataset_lr import Dataset4LR
from src.dataset_clf import Dataset4CLF
from src.dataset_lr_insurance import Dataset4LRInsurance
from src.dataset_clf_wine_quality import Dataset4CLFWineQuality
from src.decision_tree import DecisionTree
from src.saving_strategy_sl import SavingStrategy4SL
from src.plotting_strategy_lr import PlottingStrategy4LR
from src.plotting_strategy_clf import PlottingStrategy4CLF

__all__ = [
    'Polynomial',
    'LinearRegressor',
    'Dataset4SL',
    'Dataset4LR',
    'Dataset4CLF',
    'Dataset4LRInsurance',
    'Dataset4CLFWineQuality',
    'DecisionTree',
    'SavingStrategy4SL',
    'PlottingStrategy4LR',
    'PlottingStrategy4CLF'
]
