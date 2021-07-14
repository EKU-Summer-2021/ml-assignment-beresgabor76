import unittest
from src.insurance_data import InsuranceData

class InsuranceDataTest(unittest.TestCase):
    def setUp(self):
        self.__data = InsuranceData()

    def test_read_file(self):
        self.__data._InsuranceData__read_file()
        self.assertEqual((1338, 7), self.__data._InsuranceData__dataset.shape)

    def test_category_to_number(self):
        self.__data._InsuranceData__read_file()
        self.__data._InsuranceData__category_to_number()
        self.assertEqual((1338, 9), self.__data._InsuranceData__dataset_x.shape)
        self.assertEqual((1338,), self.__data._InsuranceData__dataset_y.shape)

    def test_normalize_data(self):
        self.__data._InsuranceData__read_file()
        self.__data._InsuranceData__category_to_number()
        self.__data._InsuranceData__normalize_data()
        self.__data._InsuranceData__dataset_x[(self.__data._InsuranceData__dataset_x >= 0) &
                                              (self.__data._InsuranceData__dataset_x <= 1)] = True
        all_in_interval = self.__data._InsuranceData__dataset_x.to_numpy().all()
        self.assertEqual(True, all_in_interval)

    def test_split_dataset(self):
        self.__data._InsuranceData__read_file()
        self.__data._InsuranceData__category_to_number()
        self.__data._InsuranceData__normalize_data()
        self.__data._InsuranceData__split_dataset()
        train_set_row_cnt = int(self.__data._InsuranceData__dataset_x.shape[0]
                                * (1 - self.__data._InsuranceData__test_size))
        self.assertEqual(train_set_row_cnt, self.__data.train_set_x.shape[0])
        self.assertEqual(train_set_row_cnt, self.__data.train_set_y.shape[0])
        test_set_row_cnt = 1338 - train_set_row_cnt
        self.assertEqual(test_set_row_cnt, self.__data.test_set_x.shape[0])
        self.assertEqual(test_set_row_cnt, self.__data.test_set_y.shape[0])

    def test_prepare(self):
        self.__data.prepare()
        train_set_row_cnt = int(self.__data._InsuranceData__dataset_x.shape[0]
                                * (1 - self.__data._InsuranceData__test_size))
        self.assertEqual(train_set_row_cnt, self.__data.train_set_x.shape[0])
        self.assertEqual(train_set_row_cnt, self.__data.train_set_y.shape[0])
        test_set_row_cnt = 1338 - train_set_row_cnt
        self.assertEqual(test_set_row_cnt, self.__data.test_set_x.shape[0])
        self.assertEqual(test_set_row_cnt, self.__data.test_set_y.shape[0])

if __name__ == '__main__':
    unittest.main()
