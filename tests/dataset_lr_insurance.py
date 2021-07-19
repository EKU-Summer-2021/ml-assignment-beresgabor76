import unittest
from src import Dataset4LRInsurance


class Dataset4LRInsuranceTest(unittest.TestCase):
    def setUp(self):
        self.__test_size = 0.2
        self.__data = Dataset4LRInsurance(test_size=self.__test_size)

    def test_prepare(self):
        self.__data.prepare()
        test_set_row_cnt = round(self.__data._dataset_x.shape[0] * self.__test_size)
        train_set_row_cnt = self.__data._dataset_x.shape[0] - test_set_row_cnt
        self.assertEqual(train_set_row_cnt, self.__data.train_set_x.shape[0])
        self.assertEqual(9, self.__data.train_set_x.shape[1])
        self.assertEqual(train_set_row_cnt, self.__data.train_set_y.shape[0])
        self.assertEqual(test_set_row_cnt, self.__data.test_set_x.shape[0])
        self.assertEqual(9, self.__data.test_set_x.shape[1])
        self.assertEqual(test_set_row_cnt, self.__data.test_set_y.shape[0])


if __name__ == '__main__':
    unittest.main()
