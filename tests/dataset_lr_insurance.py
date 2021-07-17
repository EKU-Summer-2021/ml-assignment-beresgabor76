import unittest
from src import Dataset4LRInsurance


class Dataset4LRInsuranceTest(unittest.TestCase):
    def setUp(self):
        self.__data = Dataset4LRInsurance(test_size=0.2)

    def test_prepare(self):
        self.__data.prepare()
        test_set_row_cnt = round(1338 * 0.2)
        train_set_row_cnt = 1338 - test_set_row_cnt
        self.assertEqual(train_set_row_cnt, self.__data.train_set_x.shape[0])
        self.assertEqual(9, self.__data.train_set_x.shape[1])
        self.assertEqual(train_set_row_cnt, self.__data.train_set_y.shape[0])
        self.assertEqual(test_set_row_cnt, self.__data.test_set_x.shape[0])
        self.assertEqual(9, self.__data.test_set_x.shape[1])
        self.assertEqual(test_set_row_cnt, self.__data.test_set_y.shape[0])


if __name__ == '__main__':
    unittest.main()
