'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from src.polynomial import Polynomial
from src.insurance_data import InsuranceData
from src.linear_regressor import LinearRegressor
from src.dataset_4SL import Dataset4SL
from src.dataset_4LR import Dataset4LR
from src.dataset_4CLF import Dataset4CLF
from src.dataset_4LR_insurance import Dataset4LRInsurance
from src.dataset_4CLF_wine_quality import Dataset4CLFWineQuality
from src.decision_tree import DecisionTree

__all__ = [
    'Polynomial',
    'InsuranceData',
    'LinearRegressor',
    'Dataset4SL',
    'Dataset4LR',
    'Dataset4CLF',
    'Dataset4LRInsurance',
    'Dataset4CLFWineQuality',
    'DecisionTree'
]
