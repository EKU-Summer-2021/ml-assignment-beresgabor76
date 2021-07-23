import unittest
import os
from src.dataset_mlp_clf_wines import Dataset4MlpClfWines
from src.saving_strategy_sl import SavingStrategy4SL
from src.plotting_strategy_clf import PlottingStrategy4CLF
from src.mlp_classifier import MlpClassifier


class MlpClassifierTest(unittest.TestCase):
    def setUp(self):
        self.__test_size = 0.2
        self.__data = Dataset4MlpClfWines(test_size=self.__test_size)
        self.__mlp_clf = MlpClassifier(SavingStrategy4SL(), PlottingStrategy4CLF())
        self.__mlp_clf.set_parameters(activation='relu', hidden_layer_sizes=(5, 10, 20, 10, 5), max_iter=5000)

    def test_train(self):
        self.__data.prepare()
        self.__mlp_clf.train(self.__data.train_set_x, self.__data.train_set_y)
        score = self.__mlp_clf._MlpClassifier__mlp.score(self.__data.test_set_x,
                                                         self.__data.test_set_y)
        self.assertEqual(True, score > 0.5)

    def test_test(self):
        self.__data.prepare()
        self.__mlp_clf.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__mlp_clf.test(self.__data.test_set_x, self.__data.test_set_y,
                            self.__data.x_scaler)
        test_set_row_cnt = round(self.__data._dataset_x.shape[0] * self.__test_size)
        self.assertEqual(test_set_row_cnt, self.__mlp_clf._prediction.shape[0])

    def test_plot_results(self):
        self.__data.prepare()
        self.__mlp_clf.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__mlp_clf.test(self.__data.test_set_x, self.__data.test_set_y,
                            self.__data.x_scaler)
        self.__mlp_clf.plot_results()
        plot_file1 = os.path.join(os.path.dirname(__file__),
                                  self.__mlp_clf._parent_dir + '/' + self.__mlp_clf._sub_dir,
                                  'histogram.png')
        self.assertEqual(True, os.path.isfile(plot_file1))
        plot_file2 = os.path.join(os.path.dirname(__file__),
                                  self.__mlp_clf._parent_dir + '/' + self.__mlp_clf._sub_dir,
                                  'confusion_mx.png')
        self.assertEqual(True, os.path.isfile(plot_file2))

    def test_save_results(self):
        self.__data.prepare()
        self.__mlp_clf.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__mlp_clf.test(self.__data.test_set_x, self.__data.test_set_y,
                            self.__data.x_scaler)
        self.__mlp_clf.save_results()
        csv_file = os.path.join(os.path.dirname(__file__),
                                self.__mlp_clf._parent_dir + '/' + self.__mlp_clf._sub_dir,
                                'results.csv')
        self.assertEqual(True, os.path.isfile(csv_file))


if __name__ == '__main__':
    unittest.main()
