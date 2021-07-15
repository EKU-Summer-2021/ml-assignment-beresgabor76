'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from src.polynomial import Polynomial
from src.insurance_data import InsuranceData
from src.linear_regressor import LinearRegressor

__all__ = [
    'Polynomial',
    'InsuranceData',
    'LinearRegressor'
]
