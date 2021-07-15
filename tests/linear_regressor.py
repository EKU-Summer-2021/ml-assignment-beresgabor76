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
        self.__regressor.save_results()
        plot_file = os.path.join(os.path.dirname(__file__),
                                 '../results/linear_regression/' +
                                 (datetime.datetime.now())
                                 .strftime('%Y-%m-%d %H:%M:%S'),
                                 'results.png')
        self.assertEqual(True, os.path.isfile(plot_file))
        modification_time = pathlib.Path(plot_file).stat().st_mtime
        self.assertEquals(True, time.time() - modification_time < 0.1)

    def test_save_results(self):
        self.__data.prepare()
        self.__regressor.train(self.__data.train_set_x, self.__data.train_set_y)
        self.__regressor.test(self.__data.test_data, self.__data.test_set_x, self.__data.test_set_y)
        self.__regressor.save_results()
        csv_file = os.path.join(os.path.dirname(__file__),
                                '../results/linear_regression/' +
                                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'results.csv')
        self.assertEqual(True, os.path.isfile(csv_file))
        modification_time = pathlib.Path(csv_file).stat().st_mtime
        self.assertEquals(True, time.time() - modification_time < 0.1)


if __name__ == '__main__':
    unittest.main()
