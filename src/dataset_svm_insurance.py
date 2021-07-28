"""
Module for class intended to store and prepare insurance data for linear regression
"""
from sklearn.preprocessing import StandardScaler
from src.dataset_nn import Dataset4NN


class Dataset4SVMInsurance(Dataset4NN):
    """
    Class intended to store and prepare insurance data for SVM regression
    """

    def __init__(self, test_size=0.2, random_state=30):
        super().__init__('insurance.csv', test_size, random_state)
        self._is_scaled_x = True
        self.x_scaler = StandardScaler()
        self._is_scaled_y = True
        self.y_scaler = StandardScaler()

    def _categories_encoding(self):
        self._category_ordinal_encoder('sex')
        self._category_ordinal_encoder('smoker')
        self._category_1hot_encoder('region')
