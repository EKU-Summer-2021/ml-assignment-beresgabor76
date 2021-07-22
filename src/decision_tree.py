"""
Module for a class for a Decision Tree Classifier
"""
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from src.learning_algorithm import LearningAlgorithm


class DecisionTree(LearningAlgorithm):
    """
    Class for training a Decision Tree Classifier model, and then test it
    """
    def __init__(self, saving_strategy, plotting_strategy):
        """
        Constructor creates a LinearRegression model
        """
        super().__init__(saving_strategy, plotting_strategy)
        self.__tree_clf = DecisionTreeClassifier()
        self._parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '../results/classification')
        self._sub_dir = self._make_save_dir()
        self._logger = self._setup_logger(f'DecisionTreeLog{self._sub_dir}',
                                          os.path.join(self._parent_dir, self._sub_dir, 'run.log'))

    def determine_hyperparameters(self, train_set_x, train_set_y):
        """
        Determines classifier hyperparameters by GridSearchCV
        """
        params = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 5, 6],
                  'min_samples_split': [10, 12, 15], 'min_samples_leaf': [7, 10, 12, 15]}
        grid_search = GridSearchCV(estimator=self.__tree_clf,
                                   param_grid=params,
                                   scoring='roc_auc_ovr',
                                   refit=True)
        grid_search.fit(train_set_x, train_set_y)
        self._logger.info('Best hyperparameters found by GridSearchCV:')
        self._logger.info(grid_search.best_params_)
        self.__tree_clf = \
            DecisionTreeClassifier(criterion=grid_search.best_params_['criterion'],
                                   max_depth=grid_search.best_params_['max_depth'],
                                   min_samples_split=grid_search.best_params_['min_samples_split'],
                                   min_samples_leaf=grid_search.best_params_['min_samples_leaf'])

    def train(self, train_set_x, train_set_y):
        """
        Trains the LinearRegression model with passed data
        """
        self.__tree_clf.fit(train_set_x, train_set_y)
        train_set_y_pred = cross_val_predict(self.__tree_clf,
                                             train_set_x,
                                             train_set_y,
                                             cv=3)
        conf_mx = confusion_matrix(train_set_y, train_set_y_pred)
        self._logger.info('Confusion matrix on train set:')
        self._logger.info('\n%s', conf_mx)

    def test(self, test_set_x, test_set_y):
        """
        Tests the LinearRegression model with passed data
        """
        self._copy_datasets(test_set_x, test_set_y)
        test_set_y_pred = self.__tree_clf.predict(test_set_x)
        self._prediction = pd.DataFrame(test_set_y_pred,
                                        index=test_set_x.index,
                                        columns=['prediction'])
        self._prediction.reset_index(inplace=True)
        self._prediction = self._prediction.drop('index', axis=1)
        conf_mx = confusion_matrix(test_set_y, test_set_y_pred)
        self._logger.info('Confusion matrix on test set:')
        self._logger.info('\n%s', conf_mx)
