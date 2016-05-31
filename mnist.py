import gzip
import pickle

import numpy as np

import conf
from theano import theano_utilities as tu


class MNIST(object):
    @staticmethod
    def convert_data_set(data, limit=None):
        x, y = data
        x = x.reshape(x.shape + (1,))
        y1 = np.zeros(y.shape + (10,))
        for i, n in enumerate(y):
            y1[i][n] = 1
        y1 = y1.reshape(y1.shape + (1,))
        if limit is not None:
            x = x[:limit]
            y1 = y1[:limit]
        return x, y1

    def __init__(self, path=conf.PATH_MNIST+'/mnist.pkl.gz'):
        with gzip.open(path) as f:
            self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding='latin1')

    def get_train_set(self, limit=None):
        return self.convert_data_set(self.train_set, limit)

    def get_valid_set(self):
        return self.convert_data_set(self.valid_set)

    def get_test_set(self):
        return self.convert_data_set(self.test_set)

    def theano_train_set(self):
        return tu.shared_dataset(self.train_set)

    def theano_valid_set(self):
        return tu.shared_dataset(self.valid_set)

    def theano_test_set(self):
        return tu.shared_dataset(self.test_set)

