import unittest
import os
import datetime
from src.dataset_lr_insurance import Dataset4LRInsurance
from src.saving_strategy_sl import SavingStrategy4SL
from src.plotting_strategy_lr import PlottingStrategy4LR
from src.mlp_regressor import MlpRegressor

class MlpRegressorTest(unittest.TestCase):
    def setUp(self):
        self.__test_size = 0.2
        self.__data = Dataset4LRInsurance(test_size=self.__test_size)
        self.__regressor = MlpRegressor(SavingStrategy4SL(), PlottingStrategy4LR())
        self.__regressor.set_parameters(activation='relu', hidden_layer_sizes=(5, 10, 20, 10, 5), max_iter=5000)

    def test_train(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        score = self.__regressor._MlpRegressor__mlp.score(self.__data.test_set_x,
                                                          self.__data.test_set_y)
        self.assertEqual(True, score > 0.75)

    def test_test(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__regressor.test(self.__data.test_set_x, self.__data.test_set_y,
                              self.__data.x_scaler)
        test_set_row_cnt = round(self.__data._dataset_x.shape[0] * self.__test_size)
        self.assertEqual(test_set_row_cnt, self.__regressor._prediction.shape[0])

    def test_plot_results(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__regressor.test(self.__data.test_set_x, self.__data.test_set_y,
                              self.__data.x_scaler)
        self.__regressor.plot_results()
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        save_path = save_time.strftime('%Y-%m-%d %H:%M:%S')
        plot_file = os.path.join(os.path.dirname(__file__),
                                 self.__regressor._parent_dir + '/' + self.__regressor._sub_dir,
                                 'results.png')
        self.assertEqual(True, os.path.isfile(plot_file))

    def test_save_results(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__regressor.test(self.__data.test_set_x, self.__data.test_set_y,
                              self.__data.x_scaler)
        self.__regressor.save_results()
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        save_path = save_time.strftime('%Y-%m-%d %H:%M:%S')
        csv_file = os.path.join(os.path.dirname(__file__),
                                self.__regressor._parent_dir + '/' + self.__regressor._sub_dir,
                                'results.csv')
        self.assertEqual(True, os.path.isfile(csv_file))


if __name__ == '__main__':
    unittest.main()
