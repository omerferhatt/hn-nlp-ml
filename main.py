import numpy as np
from data.dataset import Train, Test
from data import *

tr = Train("data/hns_2018_2019.csv", "output/vocabulary.txt", 2018, col_list, labels)
te = Test("data/hns_2018_2019.csv", "output/vocabulary.txt", 2019, col_list, labels, train=tr)
te.evaluate()
te.save_result()
