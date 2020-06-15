import sys
import argparse
import time
import numpy as np
from data.dataset import Train, Test
from data import col_list


parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, nargs=1, action="store",
                    default="data/hns_2018_2019.csv")

args = parser.parse_args()

t1 = time.time()
print("Creating baseline train model with 2018 data.")
tr_base = Train(args.data_path, "output/vocabulary.txt", 2018, col_list)
print("Predicting 2019 data with baseline model.")
te_base = Test("data/hns_2018_2019.csv", "output/vocabulary.txt", 2019, col_list, train=tr_base)
te_base.evaluate()
te_base.save_result()
print(f"Elapsed time {time.time() - t1:.2f} seconds.\n")

t1 = time.time()
print("Creating stop-words train model with 2018 data.")
tr_stop = Train("data/hns_2018_2019.csv", "output/vocabulary_stop.txt", 2018, col_list, is_stop_model=True)
print("Predicting 2019 data with stop-words model.")
te_stop = Test("data/hns_2018_2019.csv", "output/vocabulary_stop.txt", 2019, col_list, train=tr_stop, is_stop_model=True)
te_stop.evaluate()
te_stop.save_result()
print(f"Elapsed time {time.time() - t1:.2f} seconds.\n")

t1 = time.time()
print("Creating word-length train model with 2018 data.")
tr_length = Train("data/hns_2018_2019.csv", "output/vocabulary_length.txt", 2018, col_list, is_word_length_model=True)
print("Predicting 2019 data with word-length model.")
te_length = Test("data/hns_2018_2019.csv", "output/vocabulary_length.txt", 2019, col_list, train=tr_length, is_word_length_model=True)
te_length.evaluate()
te_length.save_result()
print(f"Elapsed time {time.time() - t1:.2f} seconds.\n")

t1 = time.time()
freq_list = [1, 5, 10, 15, 20]
res = []
for freq in freq_list:
    print(f"Removing frequency<={freq} from base model")
    tr_base.remove_freq(freq)
    print(f"Predicting..\n")
    tr_base.prob, tr_base.class_counts = tr_base.calc_prob()
    te = Test("data/hns_2018_2019.csv", "output/vocabulary.txt", 2019, col_list, train=tr_base)
    te.evaluate()
    temp = []
    for row in te.results:
        temp.append([row[1], row[2], row[4], row[5]])
    res.append([len(tr_base.vocabulary), temp])
res = np.array(res)

print(f"Elapsed time {time.time() - t1:.2f} seconds.\n")
print("Program terminated")
sys.exit()
