import unittest
from src.dataset_ul_students import Dataset4ULStudentsPerformance


class Dataset4ULStudentsPerformanceTest(unittest.TestCase):
    def setUp(self):
        self.__data = Dataset4ULStudentsPerformance()

    def test_prepare(self):
        self.__data.prepare()
        self.assertEqual(1000, self.__data.dataset.shape[0])
        self.assertEqual(12, self.__data.dataset.shape[1])


if __name__ == '__main__':
    unittest.main()
