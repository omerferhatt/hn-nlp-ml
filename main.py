import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data.dataset import Train, Test
from data import col_list
from copy import deepcopy

# If new dataset going to be used in this program
# Just change data path otherwise use default values
parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, nargs=1, action="store",
                    default="data/hns_2018_2019.csv")
# Parsing arguments from CLI
args = parser.parse_args()

# Starting time for baseline model
t1 = time.time()
print("Creating baseline train model with 2018 data.")
tr_base = Train(args.data_path, "output/vocabulary.txt", 2018, col_list)
print("Predicting 2019 data with baseline model.")
te_base = Test("data/hns_2018_2019.csv", "output/vocabulary.txt", 2019, col_list, train=tr_base)
te_base.evaluate()
te_base.save_result()
print(f"Elapsed time {time.time() - t1:.2f} seconds.\n")

# Starting time for stopwords model
t1 = time.time()
print("Creating stop-words train model with 2018 data.")
tr_stop = Train("data/hns_2018_2019.csv", "output/vocabulary_stop.txt", 2018, col_list, is_stop_model=True)
print("Predicting 2019 data with stop-words model.")
te_stop = Test("data/hns_2018_2019.csv", "output/vocabulary_stop.txt", 2019, col_list, train=tr_stop, is_stop_model=True)
te_stop.evaluate()
te_stop.save_result()
print(f"Elapsed time {time.time() - t1:.2f} seconds.\n")

# Starting time for word length model
t1 = time.time()
print("Creating word-length train model with 2018 data.")
tr_length = Train("data/hns_2018_2019.csv", "output/vocabulary_length.txt", 2018, col_list, is_word_length_model=True)
print("Predicting 2019 data with word-length model.")
te_length = Test("data/hns_2018_2019.csv", "output/vocabulary_length.txt", 2019, col_list, train=tr_length, is_word_length_model=True)
te_length.evaluate()
te_length.save_result()
print(f"Elapsed time {time.time() - t1:.2f} seconds.\n")

# For task 3 removing frequencies and start to calculate prob of models
tr_base = Train(args.data_path, "output/vocabulary.txt", 2018, col_list)
t1 = time.time()
freq_list = [1, 5, 10, 15, 20]
res_freq = []
for freq in freq_list:
    print(f"Removing frequency <= {freq} from base model")
    tr_base.remove_freq(freq)
    print(f"Predicting..\n")
    tr_base.prob, tr_base.class_counts = tr_base.calc_prob()
    te_freq = Test("data/hns_2018_2019.csv", "output/vocabulary.txt", 2019, col_list, train=tr_base)
    te_freq.evaluate()
    temp = []
    for row in te_freq.results:
        temp.append([row[2], row[4]])
    res_freq.append([len(tr_base.vocabulary), temp])
res_freq = np.array(res_freq)

# For task 3 removing top frequency percentages and start to calculate prob of models
tr = Train(args.data_path, "output/vocabulary.txt", 2018, col_list)
freq_perc = [5, 10, 15, 20, 25]
res_perc = []
for freq in freq_perc:
    print(f"Removing top {freq}% frequency from base model")
    tr.remove_perc_freq(freq)
    print(f"Predicting..\n")
    tr.prob, tr.class_counts = tr.calc_prob()
    te_perc = Test("data/hns_2018_2019.csv", "output/vocabulary.txt", 2019, col_list, train=tr)
    te_perc.evaluate()
    temp = []
    for row in te_perc.results:
        temp.append([row[2], row[4]])
    res_perc.append([len(tr.vocabulary), temp])
res_perc = np.array(res_perc)

# Getting results into lists
res_freq_acc = []
for f in res_freq:
    res_freq_acc.append([f[0], accuracy_score(np.array(f[1])[:, 0], np.array(f[1])[:, 1])])
res_freq_acc = np.array(res_freq_acc)

res_freq_perc_acc = []
for f in res_perc:
    res_freq_perc_acc.append([f[0], accuracy_score(np.array(f[1])[:, 0], np.array(f[1])[:, 1])])
res_freq_perc_acc = np.array(res_freq_perc_acc)

#Plotting results and saving it into output folder
plt.scatter(res_freq_acc[:, 0], res_freq_acc[:, 1])
plt.xlabel("Vocab Size")
plt.ylabel("Accruacy")
plt.savefig("output/plot_freq.png")
plt.close()
plt.scatter(res_freq_perc_acc[:, 0], res_freq_perc_acc[:, 1])
plt.xlabel("Vocab Size")
plt.ylabel("Accruacy")
plt.savefig("output/plot_perc_freq.png")
plt.close()

print(f"Elapsed time {time.time() - t1:.2f} seconds.\n")
print("Program terminated")
sys.exit()
