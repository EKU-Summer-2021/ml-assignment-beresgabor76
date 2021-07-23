import unittest
import os
from src.dataset_clf_wine_quality import Dataset4CLFWineQuality
from src.decision_tree import DecisionTree
from src.saving_strategy_sl import SavingStrategy4SL
from src.plotting_strategy_clf import PlottingStrategy4CLF


class DecisionTreeTest(unittest.TestCase):
    def setUp(self):
        self.__test_size = 0.2
        self.__data = Dataset4CLFWineQuality(test_size=self.__test_size)
        self.__tree_clf = DecisionTree(SavingStrategy4SL(), PlottingStrategy4CLF())

    def test_test(self):
        self.__data.prepare()
        self.__tree_clf.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__tree_clf.test(self.__data.test_set_x, self.__data.test_set_y)
        test_set_row_cnt = round(self.__data._dataset_x.shape[0] * self.__test_size)
        self.assertEqual(test_set_row_cnt, self.__tree_clf._prediction.shape[0])

    def test_plot_results(self):
        self.__data.prepare()
        self.__tree_clf.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__tree_clf.test(self.__data.test_set_x, self.__data.test_set_y)
        self.__tree_clf.plot_results()
        plot_file1 = os.path.join(os.path.dirname(__file__),
                                 self.__tree_clf._parent_dir + '/' + self.__tree_clf._sub_dir,
                                 'histogram.png')
        self.assertEqual(True, os.path.isfile(plot_file1))
        plot_file2 = os.path.join(os.path.dirname(__file__),
                                 self.__tree_clf._parent_dir + '/' + self.__tree_clf._sub_dir,
                                 'confusion_mx.png')
        self.assertEqual(True, os.path.isfile(plot_file2))

    def test_save_results(self):
        self.__data.prepare()
        self.__tree_clf.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__tree_clf.test(self.__data.test_set_x, self.__data.test_set_y)
        self.__tree_clf.save_results()
        csv_file = os.path.join(os.path.dirname(__file__),
                                self.__tree_clf._parent_dir + '/' + self.__tree_clf._sub_dir,
                                'results.csv')
        self.assertEqual(True, os.path.isfile(csv_file))


if __name__ == '__main__':
    unittest.main()
