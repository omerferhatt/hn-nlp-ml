# Importing main natural language processing and data pre-processing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords

# Importing custom libraries
from data import *


class Dataset:
    """
    Holds main dataset variables and dataset specifications
    """
    def __init__(self, data_path, vocab_path, col_order, save_mod=True,
                 is_stop_model=False, is_word_length_model=False, is_word_freq_model=False):
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.col_order = col_order
        self.save_mod = save_mod
        self.x_cols = ['Title', 'year']
        # Conditions for other model creations
        self.is_stop_model = is_stop_model
        self.is_word_length_model = is_word_length_model
        self.is_word_freq_model = is_word_freq_model
        # Gets data from csv file and encode classes into decimal
        self.data_frame, self.x, self.y = self.get_data()
        self.encoder, self.y = self.encode()

    def get_data(self):
        # Reading csv with specific column order. Features first, class labels at last.
        data_frame = pd.read_csv(self.data_path, usecols=self.col_order).reindex(columns=self.col_order)
        # Split data frame into training data's and their labels
        x = data_frame.iloc[:, :-1]
        y = data_frame.iloc[:, -1]
        return data_frame, x, y

    def get_tokens(self):
        # Creating set from stopwords in order to use in stop-word model
        stop_words = set(stopwords.words('english'))
        # Creating empty dictionary to hold repetition counts for words in classes
        vote_bin = {}
        for i in range(len(self.encoder.classes_)):
            # Adding class labels into dictionary and assign empty list to them.
            vote_bin[f"class {i}"] = []
        data = self.data_frame[self.x_cols].to_numpy()
        # Advance the loop simultaneously with both labels and titles,
        # and add the words to the respective classes in order.
        for cls, title_year in sorted(zip(self.y, data), key=lambda r: r[0]):
            seq = []
            # Only training model needs to get tokenize word by word without using whole title
            # Thats why we use only 2018 values in here
            if title_year[1] == 2018:
                # Get tokens from news title
                tokens = nltk.word_tokenize(title_year[0])
                for token in tokens:
                    # Testing the conditions of being a word
                    if token.isnumeric() or not token.isalpha():
                        continue

                    # Using these 2 elif conditions helps to get ask and hn words in one word like ask_hn
                    elif token.lower() == "ask" or token.lower() == "show":
                        seq += [token.lower()]
                        continue
                    elif token.lower() == "hn" and seq != []:
                        vote_bin[f"class {cls}"].append(f"{seq[0]}_hn")

                    # If stop-model wanted using this condition too with baseline model
                    elif self.is_stop_model and token.lower() in stop_words:
                        continue
                    # If word length model wanted using this condition too with baseline model
                    elif self.is_word_length_model and (len(token) >= 9 or len(token) <= 2):
                        continue
                    # Add words according to their classes
                    else:
                        vote_bin[f"class {cls}"].append(token.lower())
        # Vote bin now holds words according to their classes
        return vote_bin

    def get_title_tokens(self, year):
        # Almost same with get_tokens() function, in addition returns whole title too
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
        # Creating a vocabulary wtih whole words in every classes
        word_list = list(np.concatenate(list(self.vote_bin.values())))
        # Using set in here helps to remove duplicates
        word_list_sorted = sorted(set(np.array(word_list)), key=str.lower)
        # Saving into vocabulary file
        with open(self.vocab_path, 'w', encoding="utf-8") as f:
            for word in word_list_sorted:
                f.write(str(word) + "\n")
        return word_list_sorted

    def encode(self):
        # Encoding string classes into numerical values
        le_y = LabelEncoder()
        le_y.fit(labels)
        y = le_y.transform(self.y)
        return le_y, y

    def count_tokens(self):
        # Loop over the whole dictionary and counts how many times words repeat in classes
        count_dict = {}
        for i in range(len(self.encoder.classes_)):
            count_dict[f"class {i}"] = []
        # Get words
        words = list(np.concatenate(list(self.vote_bin.values())))
        # Sort them in A-Z
        words_sorted_unrep = sorted(set(np.array(words)), key=str.lower)
        for cls_index, cls in enumerate(list(self.vote_bin.values())):
            for word in words_sorted_unrep:
                count_dict[f'class {cls_index}'].append([cls.count(word)])
        return count_dict

    def calc_prob(self):
        # One of the most important part of Naive Bayes classifier
        # Calculates the probability of occurrence of the word in the class and
        # the whole dictionary, and the resulting probability softens the density function
        # Total word counts in every classes
        class_word_dist = np.sum(self.freq_list, axis=0)
        # Total unique words in vocabulary
        total_unique_words = len(self.vocabulary)
        # Placeholder for results
        results = []

        # Getting word and its repetition in every class
        for idx, (word, rep) in enumerate(
                zip(self.vocabulary, np.array(list(self.count_list.values())).T.squeeze())):
            temp = []
            # Looping over classes
            for i, cls in enumerate(rep):
                # Calculating probability of the word for each class in vocabulary
                prob = (cls + 10) / (class_word_dist[i] + total_unique_words)
                temp.append(prob)
            results.append(temp)
        # If 3 classes in dataset, returns like [..., [0.002, 0.001, 0.020], ... ] and word distribution
        return results, class_word_dist

    def get_word_freq(self):
        # Calculating word frequencies in vocabulary
        freq_list = np.squeeze(np.array(list(self.count_list.values()), dtype=np.int16)).T
        freq_count_list = np.array([sum(row) for row in freq_list])
        return freq_list, freq_count_list

    def save_model(self):
        # Saving model into txt file
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
    """
    Uses parent methods to create dataset
    """
    def __init__(self, data_path, vocab_path, train_year, col_order,
                 is_stop_model=False, is_word_length_model=False, is_word_freq_model=False):
        super().__init__(data_path, vocab_path, col_order, is_stop_model, is_word_length_model,
                         is_word_freq_model)
        self.train_year = train_year
        self.is_stop_model = is_stop_model
        self.is_word_length_model = is_word_length_model
        self.is_word_freq_model = is_word_freq_model
        # Get words in titles
        self.vote_bin = self.get_tokens()
        # Create vocabularu
        self.vocabulary = self.get_vocab()
        # Count words according to their classes
        self.count_list = self.count_tokens()
        # Calculate frequency
        self.freq_list, self.freq_count_list = self.get_word_freq()
        # Calculate probability
        self.prob, self.class_counts = self.calc_prob()
        # Save model into txt file
        self.save_model()

    def remove_freq(self, freq):
        # In order to doing Task 3, removing specific frequency range and updating all other variables
        # e.g. vocabulary, total word distribution etc.
        freq_count_list = np.array([sum(row) for row in self.freq_list])
        # Checks frequency conditon
        ind = np.where(freq_count_list > freq)[0]
        # Updates
        self.freq_list = np.array(self.freq_list)[ind]
        self.freq_count_list = np.array(freq_count_list)[ind]
        self.vocabulary = np.array(self.vocabulary)[ind]
        for key, value in zip(self.count_list.keys(), self.count_list.values()):
            value = np.squeeze(np.array(value))[ind]
            self.count_list[key] = value

    def remove_perc_freq(self, perc):
        # In order to doing Task 3, removing top percentage frequency and updating all other variables
        # e.g. vocabulary, total word distribution etc.
        freq_count_list = np.array([sum(row) for row in self.freq_list])
        # Checks frequency condition
        indexed = np.array([list(freq_count_list), list(np.arange(len(freq_count_list)))]).T
        srt = np.array(sorted(indexed, key=lambda x: x[0], reverse=True))
        # Updates
        ind_len = int(len(freq_count_list) / perc)
        ind = srt[ind_len:, 1]
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
        # Gets tokens, titles and classes at the same time
        self.title_tokens = self.get_title_tokens(self.test_year)
        self.results = None

    def evaluate(self):
        # Calculate probability of the title using word probabilities in train model
        # Loops over every classes and find maximum probability
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
        # If model contains 3 class it returns like [0.004, 0.002, 0.010] for every title
        self.results = results

    def save_result(self):
        # Save results in txt file
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
    # Testing functions
    print("Creating model")
    tr = Train("../data/hns_2018_2019.csv", "../output/vocabulary.txt", 2018, col_list)
    tr.remove_perc_freq(10)
