from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
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
from src.dataset_nn_insurance import Dataset4NNInsurance
from src.mlp_network import MlpNetwork
from src.dataset_mlp_clf_wines import Dataset4MlpClfWines


if __name__ == '__main__':

    dataset1 = Dataset4LRInsurance(test_size=0.2, random_state=25)
    dataset1.prepare()
    regressor = LinearRegressor(SavingStrategy4SL(), PlottingStrategy4LR())
    regressor.train(dataset1.train_set_x, dataset1.train_set_y)
    regressor.test(dataset1.test_set_x, dataset1.test_set_y, dataset1.scaler)
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
    dbscan.clustering(dataset3.dataset, dataset3.scaler)
    dbscan.save_results()
    dbscan.plot_clusters()
    dbscan.test_clustering()

    dataset4 = Dataset4LRInsurance(test_size=0.2, random_state=25)
    dataset4.prepare()
    regressor = MlpRegressor(SavingStrategy4SL(), PlottingStrategy4LR())
    regressor.set_parameters(activation='relu', hidden_layer_sizes=(5, 10, 20, 10, 5), max_iter=10000)
    #regressor.determine_parameters(dataset4.train_set_x, dataset4.train_set_y)
    regressor.train(dataset4.train_set_x, dataset4.train_set_y)
    regressor.test(dataset4.test_set_x, dataset4.test_set_y, dataset4.x_scaler)
    regressor.plot_results()
    regressor.save_results()

    dataset5 = Dataset4NNInsurance(test_size=0.2, random_state=25)
    dataset5.prepare()
    regressor = MlpNetwork(MLPRegressor(), SavingStrategy4SL(), PlottingStrategy4LR())
    regressor.set_parameters(activation='tanh',
                             hidden_layer_sizes=(25, 50, 100, 50, 25),
                             learning_rate='adaptive',
                             max_iter=5000)
    #regressor.determine_parameters(dataset5.train_set_x, dataset5.train_set_y)
    regressor.train(dataset5.train_set_x, dataset5.train_set_y)
    regressor.test(dataset5.test_set_x, dataset5.test_set_y, dataset5.x_scaler, dataset5.y_scaler)
    regressor.plot_results()
    regressor.save_results()

    dataset6 = Dataset4MlpClfWines(test_size=0.2, random_state=25)
    dataset6.prepare()
    mlp_clf = MlpNetwork(MLPClassifier(), SavingStrategy4SL(), PlottingStrategy4CLF())
    mlp_clf.set_parameters(activation='relu',
                           hidden_layer_sizes=(25, 50, 100, 50, 25),
                           learning_rate='adaptive',
                           max_iter=5000)
    #mlp_clf.determine_parameters(dataset6.train_set_x, dataset6.train_set_y)
    mlp_clf.train(dataset6.train_set_x, dataset6.train_set_y)
    mlp_clf.test(dataset6.test_set_x, dataset6.test_set_y, dataset6.x_scaler)
    mlp_clf.plot_results()
    mlp_clf.save_results()
