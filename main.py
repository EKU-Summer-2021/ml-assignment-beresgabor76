from src import Dataset4LRInsurance
from src import SavingStrategy4SL
from src import PlottingStrategy4LR
from src import LinearRegressor
from src import Dataset4CLFWineQuality
from src import DecisionTree
from src import PlottingStrategy4CLF


if __name__ == '__main__':
    dataset1 = Dataset4LRInsurance(test_size=0.2, random_state=20)
    dataset1.prepare()
    dataset1.print_correlation()
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


