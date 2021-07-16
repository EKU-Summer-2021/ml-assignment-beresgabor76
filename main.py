from src import InsuranceData
from src import LinearRegressor

if __name__ == '__main__':
    dataset = InsuranceData(test_size=0.2, random_state=20)
    dataset.prepare()
    dataset.print_correlation()
    regressor = LinearRegressor()
    regressor.train(dataset.train_set_x, dataset.train_set_y)
    regressor.test(dataset.test_data, dataset.test_set_x, dataset.test_set_y)
    regressor.plot_results()
    regressor.save_results()


