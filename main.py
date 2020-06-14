import numpy as np
from data.dataset import Train, Test
from data import *

tr_base = Train("data/hns_2018_2019.csv", "output/vocabulary_base.txt", 2018, col_list)
te_base = Test("data/hns_2018_2019.csv", "output/vocabulary_base.txt", 2019, col_list, train=tr_base)
te_base.evaluate()
te_base.save_result()

tr_stop = Train("data/hns_2018_2019.csv", "output/vocabulary_stop.txt", 2018, col_list, is_stop_model=True)
te_stop = Test("data/hns_2018_2019.csv", "output/vocabulary_stop.txt", 2019, col_list, train=tr_stop, is_stop_model=True)
te_stop.evaluate()
te_stop.save_result()

tr_length = Train("data/hns_2018_2019.csv", "output/vocabulary_length.txt", 2018, col_list, is_word_length_model=True)
te_length = Test("data/hns_2018_2019.csv", "output/vocabulary_length.txt", 2019, col_list, train=tr_length, is_word_length_model=True)
te_length.evaluate()
te_length.save_result()
