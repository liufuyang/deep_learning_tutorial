import string
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical

import re


def clean(s):
    """
    Remove rare characters and change everything to lowercase
    """
    s = re.sub(r'[^\x00-\x7f]', r'', s)
    # s = re.sub("[^a-zA-Z]+", ' ', s)   # remove numbers
    return s.lower()


def load_ag_data():

    # for train data
    train = pd.read_csv('data/train.csv', header=None)
    train = train.dropna()

    x_train = train[0]
    x_train = x_train.apply(clean)
    x_train = np.array(x_train)

    y_train_unspsc = train[1]
    label_array = list(y_train_unspsc.sort_values().unique())
    y_train = y_train_unspsc.apply(lambda c: label_array.index(c))
    y_train = to_categorical(y_train)

    # for test data

    test = pd.read_csv('data/test.csv', header=None)
    x_test = test[0]
    x_test = x_test.apply(clean)
    x_test = np.array(x_test)

    y_test_unspsc = test[1]
    y_test = y_test_unspsc.apply(lambda c: label_array.index(c))
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test), label_array


def encode_data(x, maxlen, vocab):
    # Iterate over the loaded data and create a 1-D tensor of size maxlen
    # In this case that will be 200. This is then placed in a 3D matrix of size
    # data_samples x maxlen. Each character is encoded into a one-hot
    # array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen), dtype=np.int)
    for dix, sent in enumerate(x):
        counter = 0

        for c in sent:
            if counter >= maxlen:
                pass
            else:
                ix = vocab.get(c, -1)  # get index from vocab dictionary, if not in vocab, return -1
                input_data[dix, counter] = ix
                counter += 1

    return input_data


def create_vocab_set():
    # This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n', ' '])
    # alphabet = set(alphabet)
    vocab_size = len(alphabet)

    vocab = {} # dictionary, key as characters, values as index of a char in check
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, alphabet
