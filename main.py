import numpy as np
from src import InsuranceData

if __name__ == '__main__':
    dataset = InsuranceData()
    dataset.prepare()
    dataset.print_correlation()