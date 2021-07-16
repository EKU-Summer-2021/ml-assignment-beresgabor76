import unittest
import os
import datetime
import time
import pathlib
from src import InsuranceData
from src import LinearRegressor


class LinearRegressorTest(unittest.TestCase):
    def setUp(self):
        self.__data = InsuranceData()
        self.__regressor = LinearRegressor()

    def test_train(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        score = self.__regressor._LinearRegressor__lin_reg.score(self.__data.test_set_x,
                                                                 self.__data.test_set_y)
        self.assertEqual(True, score > 0.75)

    def test_test(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__regressor.test(self.__data.test_data, self.__data.test_set_x, self.__data.test_set_y)
        test_set_row_cnt = round(self.__data._InsuranceData__dataset.shape[0]
                                 * (self.__data._InsuranceData__test_size))
        self.assertEqual(test_set_row_cnt, self.__regressor._LinearRegressor__prediction.shape[0])

    def test_plot_results(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__regressor.test(self.__data.test_data, self.__data.test_set_x, self.__data.test_set_y)
        self.__regressor.plot_results()
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        save_path = save_time.strftime('%Y-%m-%d %H:%M:%S')
        plot_file = os.path.join(os.path.dirname(__file__),
                                 '../results/linear_regression/' +
                                 save_path,
                                 'results.png')
        self.assertEqual(True, os.path.isfile(plot_file))

    def test_save_results(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__regressor.test(self.__data.test_data, self.__data.test_set_x, self.__data.test_set_y)
        self.__regressor.save_results()
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        save_path = save_time.strftime('%Y-%m-%d %H:%M:%S')
        csv_file = os.path.join(os.path.dirname(__file__),
                                '../results/linear_regression/' +
                                save_path,
                                'results.csv')
        self.assertEqual(True, os.path.isfile(csv_file))


if __name__ == '__main__':
    unittest.main()