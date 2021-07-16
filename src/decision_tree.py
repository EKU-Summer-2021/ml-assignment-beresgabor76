"""
Module
"""
import os
import datetime
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


class DecisionTree:
    """
    Class for training a Linear Regression model, and then test it
    """
    def __init__(self):
        """
        Constructor creates a LinearRegression model
        """
        self.__tree_clf = DecisionTreeClassifier()
        self.__test_set = None
        self.__target = None
        self.__prediction = None
        self.__parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '../results/classification')
        self.__confusion_mx = None

    def determine_hyperparameters(self, train_set_x, train_set_y):
        params = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 5, 6],
                  'min_samples_split': [10, 12, 15], 'min_samples_leaf': [7, 10, 12, 15]}
        grid_search = GridSearchCV(estimator=self.__tree_clf,
                                   param_grid=params,
                                   scoring='roc_auc_ovr',
                                   refit=True)
        grid_search.fit(train_set_x, train_set_y)
        print('Best hyperparameters found by GridSearchCV:')
        print(grid_search.best_params_)
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
        print('Confusion matrix on train set:')
        print(conf_mx)

    def test(self, test_set_x, test_set_y):
        """
        Tests the LinearRegression model with passed data
        """
        self.__test_set = pd.DataFrame(test_set_x.copy())
        self.__test_set.reset_index(inplace=True)
        self.__test_set = self.__test_set.drop('index', axis=1)
        self.__target = pd.DataFrame(test_set_y)
        self.__target.reset_index(inplace=True)
        self.__target = self.__target.drop('index', axis=1)
        test_set_y_pred = self.__tree_clf.predict(test_set_x)
        self.__prediction = pd.DataFrame(test_set_y_pred,
                                         index=test_set_x.index,
                                         columns=['prediction'])
        self.__prediction.reset_index(inplace=True)
        self.__prediction = self.__prediction.drop('index', axis=1)
        self.__confusion_mx = confusion_matrix(test_set_y, test_set_y_pred)
        print('Confusion matrix on test set:')
        print(self.__confusion_mx)

    def plot_results(self):
        """
        Plots out the tree as well as confusion matrix fpr the test
        """
        fig = plt.figure()
        plt.hist(pd.concat([self.__target, self.__prediction], axis=1),
                 color=['blue', 'red'], label=['Actual', 'Prediction'],
                 histtype='bar', bins=6)
        plt.legend()
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        save_path = save_time.strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), self.__parent_dir, save_path)):
            os.chdir(self.__parent_dir)
            os.mkdir(save_path)
        plot_file = os.path.join(os.path.dirname(__file__), self.__parent_dir,
                                 save_path, 'results.png')
        fig.savefig(plot_file)

    def save_results(self):
        """
        Saves test dataset with target values and prediction results with errors
        """
        results_df = pd.concat([self.__test_set,
                                self.__target,
                                self.__prediction], axis=1)
        results_df['error'] = results_df.iloc[:, -1] - results_df.iloc[:, -2]
        results_df = results_df.round(2)
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
            - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        save_path = save_time.strftime('%Y-%m-%d %H:%M:%S')
        if not os.path.exists(os.path.join(os.path.dirname(__file__), self.__parent_dir, save_path)):
            os.chdir(self.__parent_dir)
            os.mkdir(save_path)
        csv_file = os.path.join(os.path.dirname(__file__), self.__parent_dir, save_path, 'results.csv')
        results_df.to_csv(path_or_buf=csv_file, index=False)