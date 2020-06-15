# Importing custom libraries
from data import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from decimal import Decimal


class Dataset:
    def __init__(self, data_path, vocab_path, col_order, save_mod=True,
                 is_stop_model=False, is_word_length_model=False, is_word_freq_model=False):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.col_order = col_order
        self.save_mod = save_mod

        self.is_stop_model = is_stop_model
        self.is_word_length_model = is_word_length_model
        self.is_word_freq_model = is_word_freq_model
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
        vote_bin = {}
        for i in range(len(self.encoder.classes_)):
            vote_bin[f"class {i}"] = []
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
                    elif token.lower() == "ask" or token.lower() == "show":
                        seq += [token.lower()]
                        continue
                    elif token.lower() == "hn" and seq != []:
                        vote_bin[f"class {cls}"].append(f"{seq[0]}_hn")
                    elif self.is_stop_model and token.lower() in stop_words:
                        continue
                    elif self.is_word_length_model and (len(token) >= 9 or len(token) <= 2):
                        continue
                    else:
                        vote_bin[f"class {cls}"].append(token.lower())
        return vote_bin

    def get_title_tokens(self, year):
        stop_words = set(stopwords.words('english'))
        data = self.data_frame[self.x_cols].to_numpy()
        titles_labels = []
        for index, (cls, title_year) in enumerate(zip(self.y, data)):
            seq = []
            title = []
            if title_year[1] == year:
                tokens = nltk.word_tokenize(title_year[0])
                for token in tokens:
                    # Testing the conditions of being a word
                    if token.isnumeric() or not token.isalpha():
                        continue
                    elif token.lower() == "ask" or token.lower() == "show":
                        seq += [token.lower()]
                        continue
                    elif token.lower() == "hn" and seq != []:
                        title.append(f"{seq[0]}_hn")
                    elif self.is_stop_model and token.lower() in stop_words:
                        continue
                    elif self.is_word_length_model and (len(token) >= 9 or len(token) <= 2):
                        continue
                    else:
                        title.append(token.lower())
                titles_labels.append([index, title_year[0], title, cls])
        return titles_labels

    def get_vocab(self):
        word_list = list(np.concatenate(list(self.vote_bin.values())))
        word_list_sorted = sorted(set(np.array(word_list)), key=str.lower)
        with open(self.vocab_path, 'w', encoding="utf-8") as f:
            for word in word_list_sorted:
                f.write(str(word) + "\n")
        return word_list_sorted

    def encode(self):
        le_y = LabelEncoder()
        le_y.fit(labels)
        y = le_y.transform(self.y)
        return le_y, y

    def count_tokens(self):
        count_dict = {}
        for i in range(len(self.encoder.classes_)):
            count_dict[f"class {i}"] = []
        words = list(np.concatenate(list(self.vote_bin.values())))
        words_sorted_unrep = sorted(set(np.array(words)), key=str.lower)
        for cls_index, cls in enumerate(list(self.vote_bin.values())):
            for word in words_sorted_unrep:
                count_dict[f'class {cls_index}'].append([cls.count(word)])
        return count_dict

    def calc_prob(self):
        class_word_dist = np.sum(self.freq_list, axis=0)
        total_used_words = np.sum(class_word_dist)
        total_unique_words = len(self.vocabulary)
        results = []

        for idx, (word, rep) in enumerate(
                zip(self.vocabulary, np.array(list(self.count_list.values())).T.squeeze())):
            temp = []
            st = np.std(class_word_dist)
            for i, cls in enumerate(rep):
                prob = (cls + 1) / (class_word_dist[i] + total_unique_words)
                temp.append(prob)
            results.append(temp)
        return results, class_word_dist

    def get_word_freq(self):
        freq_list = np.squeeze(np.array(list(self.count_list.values()), dtype=np.int16)).T
        freq_count_list = np.array([sum(row) for row in freq_list])
        return freq_list, freq_count_list

    def save_model(self):
        if self.is_stop_model:
            path = "output/stopword-model.txt"
        elif self.is_word_length_model:
            path = "output/wordlength-model.txt"
        elif self.is_word_freq_model:
            path = "output/wordfreq-model.txt"
        else:
            path = "output/model-2018.txt"
        with open(path, "w", encoding="utf-8") as f:
            for index, (word, prob) in enumerate(zip(self.vocabulary, self.prob)):
                f.write(f"{index}  {word}  ")
                for cls, p in enumerate(prob):
                    f.write(f"{self.freq_list[index, cls]:.2f}  {p:.3e}  ")
                f.write("\n")


class Train(Dataset):
    def __init__(self, data_path, vocab_path, train_year, col_order,
                 is_stop_model=False, is_word_length_model=False, is_word_freq_model=False):
        super().__init__(data_path, vocab_path, col_order, is_stop_model, is_word_length_model,
                         is_word_freq_model)
        self.train_year = train_year
        self.is_stop_model = is_stop_model
        self.is_word_length_model = is_word_length_model
        self.is_word_freq_model = is_word_freq_model
        self.vote_bin = self.get_tokens()
        self.vocabulary = self.get_vocab()
        self.count_list = self.count_tokens()
        self.freq_list, self.freq_count_list = self.get_word_freq()
        self.prob, self.class_counts = self.calc_prob()
        self.save_model()

    def remove_freq(self, freq):
        freq_count_list = np.array([sum(row) for row in self.freq_list])
        ind = np.where(freq_count_list > freq)[0]
        self.freq_list = np.array(self.freq_list)[ind]
        self.freq_count_list = np.array(freq_count_list)[ind]
        self.vocabulary = np.array(self.vocabulary)[ind]
        for key, value in zip(self.count_list.keys(), self.count_list.values()):
            value = np.squeeze(np.array(value))[ind]
            self.count_list[key] = value


class Test(Dataset):
    def __init__(self, data_path, vocab_path, test_year, col_order, train,
                 is_stop_model=False, is_word_length_model=False, is_word_freq_model=False):
        super().__init__(data_path, vocab_path, col_order, is_stop_model, is_word_length_model,
                         is_word_freq_model)
        self.train = train
        self.test_year = test_year
        self.is_stop_model = is_stop_model
        self.is_word_length_model = is_word_length_model
        self.is_word_freq_model = is_word_freq_model
        self.title_tokens = self.get_title_tokens(self.test_year)
        self.results = None

    def evaluate(self):
        results = []
        for index, t_l in enumerate(self.title_tokens):
            cls_prb = []
            for cls in range(len(self.encoder.classes_)):
                token_prob_list = []
                for token in t_l[2]:
                    if token in self.train.vocabulary:
                        token_prob_list.append(self.train.prob[list(self.train.vocabulary).index(token)][cls])
                cls_prb.append(self.train.class_counts[cls] / sum(self.train.class_counts) * np.prod(token_prob_list))
            results.append([index,
                            t_l[1],
                            t_l[3],
                            cls_prb,
                            cls_prb.index(max(cls_prb)),
                            cls_prb.index(max(cls_prb)) == t_l[3]])
        self.results = results

    def save_result(self):
        if self.is_stop_model:
            path = "output/stopword-result.txt"
        elif self.is_word_length_model:
            path = "output/wordlength-result.txt"
        elif self.is_word_freq_model:
            path = "output/wordfreq-result.txt"
        else:
            path = "output/baseline-result.txt"
        with open(path, 'w', encoding='utf-8') as f:
            for result in self.results:
                f.write(f"{result[0]}  ")
                f.write(f"{result[1]}  ")
                f.write(f"{self.train.encoder.inverse_transform((result[2],))[0]}  ")
                for i in range(len(result[3])):
                    f.write(f"{result[3][i]:.3e}  ")
                f.write(f"{self.train.encoder.inverse_transform((result[4],))[0]}  ")
                f.write(f"{'right' if result[5] == True else 'wrong'}\n")


if __name__ == "__main__":
    print("Creating model")
    tr = Train("../data/hns_2018_2019.csv", "../output/vocabulary.txt", 2018, col_list)
