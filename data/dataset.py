# Importing custom libraries
from data import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords


class Dataset:
    def __init__(self, data_path, vocab_path, col_order, label, save_mod=True):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.col_order = col_order
        self.labels = label
        self.save_mod = save_mod
        self.x_cols = ['Title', 'year']

        self.data_frame, self.x, self.y = self.get_data()
        self.encoder, self.y = self.encode()

    def get_data(self):
        data_frame = pd.read_csv(self.data_path, usecols=self.col_order).reindex(columns=self.col_order)
        # Split data frame into training data's and their labels
        x = data_frame.iloc[:, :-1]
        y = data_frame.iloc[:, -1]
        return data_frame, x, y

    def get_tokens(self):
        stop_words = set(stopwords.words('english'))
        vote_bin = {"class 0": [], "class 1": [], "class 2": []}
        data = self.data_frame[self.x_cols].to_numpy()
        # Advance the loop simultaneously with both labels and titles,
        # and add the words to the respective classes in order.
        for cls, title_year in sorted(zip(self.y, data), key=lambda r: r[0]):
            seq = []
            if title_year[1] == 2018:
                tokens = nltk.word_tokenize(title_year[0])
                for token in tokens:
                    # Testing the conditions of being a word
                    if token.isnumeric() or not token.isalpha():
                        continue
                    elif token.lower() in stop_words:
                        continue
                    elif token.lower() == "ask" or token.lower() == "show":
                        seq += [token.lower()]
                        continue
                    elif token.lower() == "hn" and seq != []:
                        vote_bin[f"class {cls}"].append(f"{seq[0]}_hn")
                    else:
                        vote_bin[f"class {cls}"].append(token.lower())
        return vote_bin

    def get_title_tokens(self, year):
        stop_words = set(stopwords.words('english'))
        data = self.data_frame[self.x_cols].to_numpy()
        # Advance the loop simultaneously with both labels and titles,
        # and add the words to the respective classes in order.

        titles_labels = []
        for cls, title_year in sorted(zip(self.y, data), key=lambda r: r[0]):
            seq = []
            title = []
            if title_year[1] == year:
                tokens = nltk.word_tokenize(title_year[0])
                for token in tokens:
                    # Testing the conditions of being a word
                    if token.isnumeric() or not token.isalpha():
                        continue
                    elif token.lower() in stop_words:
                        continue
                    elif token.lower() == "ask" or token.lower() == "show":
                        seq += [token.lower()]
                        continue
                    elif token.lower() == "hn" and seq != []:
                        title.append(f"{seq[0]}_hn")
                    else:
                        title.append(token.lower())
                titles_labels.append([title, cls])
        return titles_labels

    def get_vocab(self):
        word_list = list(np.concatenate(list(self.vote_bin.values())))
        word_list_sorted = sorted(set(np.array(word_list)), key=str.lower)
        f = open(self.vocab_path, 'w')
        for word in word_list_sorted:
            f.write(str(word) + "\n")
        f.close()
        return word_list_sorted

    def encode(self):
        le_y = LabelEncoder()
        le_y.fit(labels)
        y = le_y.transform(self.y)
        return le_y, y

    def count_tokens(self):
        count_dict = {"class 0": [], "class 1": [], "class 2": [], "class 3": []}
        words = list(np.concatenate(list(self.vote_bin.values())))
        words_sorted_unrep = sorted(set(np.array(words)), key=str.lower)
        for cls_index, cls in enumerate(list(self.vote_bin.values())):
            for word in words_sorted_unrep:
                count_dict[f'class {cls_index}'].append([cls.count(word)])
        return count_dict

    def calc_prob(self):
        class_word_dist = [len(l) for l in list(self.vote_bin.values())]
        total_used_words = sum(class_word_dist)
        total_unique_words = len(self.vocabulary)
        results = []

        for idx, (word, rep) in enumerate(
                zip(self.vocabulary, np.array(list(self.count_list.values())[:][0:-1]).T.squeeze())):
            temp = []
            for i, cls in enumerate(rep):
                prob = (cls + 1000) / (class_word_dist[i] + total_unique_words)
                temp.append(prob)
            results.append(temp)
        return results, class_word_dist


class Train(Dataset):
    def __init__(self, data_path, vocab_path, train_year, col_order, label):
        super().__init__(data_path, vocab_path, col_order, label)
        self.train_year = train_year
        self.vote_bin = self.get_tokens()
        self.vocabulary = self.get_vocab()
        self.count_list = self.count_tokens()
        self.prob, self.class_counts = self.calc_prob()


class Test(Dataset):
    def __init__(self, data_path, vocab_path, test_year, col_order, label):
        super().__init__(data_path, vocab_path, col_order, label)
        self.test_year = test_year
        self.title_tokens = self.get_title_tokens(self.test_year)

    def evaluate(self, train):
        results = []
        for t_l in self.title_tokens:
            cls_prb = []
            for cls in range(3):
                token_prob_list = []
                for token in t_l[0]:
                    if token in train.vocabulary:
                        token_prob_list.append(train.prob[train.vocabulary.index(token)][cls])
                cls_prb.append(train.class_counts[cls] / sum(train.class_counts) * np.prod(token_prob_list))
            results.append([t_l[0], t_l[1], cls_prb.index(max(cls_prb))])
        return results
