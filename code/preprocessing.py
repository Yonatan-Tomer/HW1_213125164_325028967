from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple
from matplotlib import pyplot as plt

WORD = 0
TAG = 1

FEATURE_CLASSES = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "capital", "number",
                   "contains hyphen", "pp_word", "f100_lower", "f101_lower", "f102_lower", "f106_lower",
                   "f107_lower", "nn_word", "nn_word_lower", "c_word_n_word", "length", "2words", "3words"]


class FeatureStatistics:
    def __init__(self):
        # self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        # the feature classes used in ,the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in FEATURE_CLASSES}

        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 2):
                    history = (
                        sentence[i][WORD], sentence[i][TAG], sentence[i - 1][WORD], sentence[i - 1][TAG],
                        sentence[i - 2][WORD],
                        sentence[i - 2][TAG], sentence[i + 1][WORD], sentence[i + 2][WORD])
                    # extract features from history and count them
                    data_class_pairs = history_to_data_and_class(history)
                    for feature_data, feature_class in data_class_pairs:
                        self.count_feature_data(feature_data, feature_class)

                    self.histories.append(history)

    def count_feature_data(self, feature_data, feature_class):
        """
        Count a feature appearance
        """
        if feature_data not in self.feature_rep_dict[feature_class]:
            self.feature_rep_dict[feature_class][feature_data] = 1
        else:
            self.feature_rep_dict[feature_class][feature_data] += 1




class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {}
        for feature_class in FEATURE_CLASSES:
            self.feature_to_idx[feature_class] = OrderedDict()
        self.represent_input_with_features = OrderedDict()
        self.h = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6], hist[7])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def history_to_data_and_class(history: Tuple):
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word, nn_word = history
    data_class_pairs = [
        ((c_word, c_tag), "f100"),
        ((pp_tag, p_tag, c_tag), "f103"),
        ((p_tag, c_tag), "f104"),
        (c_tag, "f105"),
        ((p_word, c_tag), "f106"),
        ((n_word, c_tag), "f107"),
        ((any(ch.isdigit() for ch in c_word), c_tag), "number"),
        ((any(ch.isupper() for ch in c_word), c_tag), "capital"),
        (('-' in c_word, c_tag), "contains hyphen"),
        ((pp_word, c_tag), "pp_word"),
        ((c_word.lower(), c_tag), "f100_lower"),
        ((p_word.lower(), c_tag), "f106_lower"),
        ((n_word.lower(), c_tag), "f107_lower"),
        ((nn_word, c_tag), "nn_word"),
        ((nn_word.lower(), c_tag), "nn_word_lower"),
        ((c_word, n_word, c_tag), "c_word_n_word"),
        ((len(c_word), c_tag), "length"),
        ((c_word, p_word, c_tag), "2words"),
        ((c_word, p_word, pp_word, c_tag), "3words"),
    ]
    word_len = len(c_word)
    for i in range(0, min(word_len, 4)):
        data_class_pairs.append(((c_word[word_len - i - 1:], c_tag), "f101"))
        data_class_pairs.append(((c_word[:i + 1], c_tag), "f102"))

        data_class_pairs.append(((c_word[word_len - i - 1:].lower(), c_tag), "f101_lower"))
        data_class_pairs.append(((c_word[:i + 1].lower(), c_tag), "f102_lower"))
    return data_class_pairs


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) \
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all ids features that are relevant to the given history
    """
    features = []
    data_class_pairs = history_to_data_and_class(history)
    for feature_data, feature_class in data_class_pairs:
        append_idx_if_exists(feature_data, feature_class, dict_of_dicts, features)

    return features


def append_idx_if_exists(feature_data, feature_class: str, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]],
                         features):
    """
    Append the index of the feature to features if it exists in dict_of_dicts[feature_class]
    """
    if feature_data in dict_of_dicts[feature_class]:
        features.append(dict_of_dicts[feature_class][feature_data])


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
