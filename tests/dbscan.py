import unittest
import os
import datetime
from src.dataset_ul_students import Dataset4ULStudentsPerformance
from src.dbscan import DbscanClustering
from src.saving_strategy_ul import SavingStrategy4UL
from src.plotting_strategy_clu import PlottingStrategy4CLU


class DbscanClusteringTest(unittest.TestCase):
    def setUp(self):
        self.__test_size = 0.2
        self.__data = Dataset4ULStudentsPerformance()
        self.__dbscan = DbscanClustering(eps=1.2, min_samples=17,
                                         saving_strategy=SavingStrategy4UL(),
                                         plotting_strategy=PlottingStrategy4CLU())

    def test_clustering(self):
        self.__data.prepare()
        self.__dbscan.clustering(self.__data.unscaled_dataset, self.__data.dataset)
        self.assertEqual(False, self.__dbscan._prediction.empty)

    def test_plot_clusters(self):
        self.__data.prepare()
        self.__dbscan.clustering(self.__data.unscaled_dataset, self.__data.dataset)
        self.__dbscan.plot_clusters()
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        save_path = save_time.strftime('%Y-%m-%d %H:%M:%S')
        plot_file = os.path.join(os.path.dirname(__file__),
                                 self.__dbscan._parent_dir + '/' + self.__dbscan._sub_dir,
                                 'clusters.png')
        self.assertEqual(True, os.path.isfile(plot_file))

    def test_save_results(self):
        self.__data.prepare()
        self.__dbscan.clustering(self.__data.unscaled_dataset, self.__data.dataset)
        self.__dbscan.plot_clusters()
        self.__dbscan.save_results()
        resolution = datetime.timedelta(seconds=10)
        save_time = datetime.datetime.now() \
                    - datetime.timedelta(seconds=datetime.datetime.now().second % resolution.seconds)
        save_path = save_time.strftime('%Y-%m-%d %H:%M:%S')
        csv_file = os.path.join(os.path.dirname(__file__),
                                self.__dbscan._parent_dir + '/' + self.__dbscan._sub_dir,
                                'results.csv')
        self.assertEqual(True, os.path.isfile(csv_file))


if __name__ == '__main__':
    unittest.main()
