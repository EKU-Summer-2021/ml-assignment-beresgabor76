import unittest
from src import Dataset4CLFWineQuality


class Dataset4CLFWineQualityTest(unittest.TestCase):
    def setUp(self):
        self.__data = Dataset4CLFWineQuality(test_size=0.2)

    def test_prepare(self):
        self.__data.prepare()
        test_set_row_cnt = round(1599 * 0.2)
        train_set_row_cnt = 1599 - test_set_row_cnt
        self.assertEqual(train_set_row_cnt, self.__data.train_set_x.shape[0])
        self.assertEqual(11, self.__data.train_set_x.shape[1])
        self.assertEqual(train_set_row_cnt, self.__data.train_set_y.shape[0])
        self.assertEqual(test_set_row_cnt, self.__data.test_set_x.shape[0])
        self.assertEqual(11, self.__data.test_set_x.shape[1])
        self.assertEqual(test_set_row_cnt, self.__data.test_set_y.shape[0])


if __name__ == '__main__':
    unittest.main()
