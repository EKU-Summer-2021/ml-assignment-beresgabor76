from src.dataset_lr_insurance import Dataset4LRInsurance
from src.saving_strategy_sl import SavingStrategy4SL
from src.plotting_strategy_lr import PlottingStrategy4LR
from src.linear_regressor import LinearRegressor
from src.dataset_clf_wine_quality import Dataset4CLFWineQuality
from src.decision_tree import DecisionTree
from src.plotting_strategy_clf import PlottingStrategy4CLF
from src.dataset_ul_students import Dataset4ULStudentsPerformance
from src.dbscan import DbscanClustering
from src.saving_strategy_ul import SavingStrategy4UL
from src.plotting_strategy_clu import PlottingStrategy4CLU

if __name__ == '__main__':
    dataset1 = Dataset4LRInsurance(test_size=0.2, random_state=25)
    dataset1.prepare()
    regressor = LinearRegressor(SavingStrategy4SL(), PlottingStrategy4LR())
    regressor.train(dataset1.train_set_x, dataset1.train_set_y)
    regressor.test(dataset1.test_data, dataset1.test_set_x, dataset1.test_set_y)
    regressor.plot_results()
    regressor.save_results()

    dataset2 = Dataset4CLFWineQuality(test_size=0.2, random_state=20)
    dataset2.prepare()
    tree_clf = DecisionTree(SavingStrategy4SL(), PlottingStrategy4CLF())
    tree_clf.determine_hyperparameters(dataset2.train_set_x, dataset2.train_set_y)
    tree_clf.train(dataset2.train_set_x, dataset2.train_set_y)
    tree_clf.test(dataset2.test_set_x, dataset2.test_set_y)
    tree_clf.plot_results()
    tree_clf.save_results()

    dataset3 = Dataset4ULStudentsPerformance()
    dataset3.prepare()
    dbscan = DbscanClustering(eps=1.2, min_samples=17,
                              saving_strategy=SavingStrategy4UL(),
                              plotting_strategy=PlottingStrategy4CLU())
    dbscan.clustering(dataset3.unscaled_dataset, dataset3.dataset)
    dbscan.save_results()
    dbscan.plot_clusters()
    dbscan.test_clustering()
