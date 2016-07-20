import os
import pickle
import sys
import numpy as np

import base_layer
import conf


class BaseNet(object):
    @staticmethod
    def list_of_same_length(l):
        return [0] * len(l)

    @staticmethod
    def _get_mini_batch(data_set, batch_size, batch_index):
        return data_set[batch_index * batch_size: (batch_index + 1) * batch_size]

    @staticmethod
    def get_batch_number(data_set, batch_size):
        return len(data_set[0]) // batch_size

    @staticmethod
    def init_neuron_layers(layer_types: []):
        raise NotImplementedError

    @staticmethod
    def init_layers(inputs_shape, layer_types):
        input_layer = base_layer.DirectLayer(inputs_shape=inputs_shape)
        layers = [input_layer] + BaseNet.init_neuron_layers(layer_types)
        return layers

    @classmethod
    def net_from_layer_types(cls, inputs_shape, layer_types):
        input_layer = base_layer.DirectLayer(inputs_shape=inputs_shape)
        layers = [input_layer] + cls.init_neuron_layers(layer_types)
        net = cls(layers)
        return net

    @classmethod
    def init_with_train_model(cls, layers,
                              train_set, batch_size, n_epochs, learning_rate, cost_func, l1_a=0.0, l2_a=0.0001):
        """
        :parameter layers: a list of tuple for init layers
        """
        net = cls(layers)
        net.train(train_set, batch_size, n_epochs, learning_rate, cost_func, l1_a, l2_a)
        return net

    def __init__(self, layers: [base_layer.AbstractLayer]):
        """
        :parameter layers: a list of tuple for init layers
        """

        self.layers = layers
        self.weighted_layers = self.get_weighted_layers()
        self.rng = np.random.RandomState(1234)
        self.depth = len(self.weighted_layers)

        # set pickle file path
        self.file_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.pickle_file = conf.DATA_PATH + self.file_name + '.pkl'

        self.connect()
        # self.reset_params()
        self.init_weights_baises()

        # l1 and l2 regularization
        self.l1, self.l2 = self.init_regularization()

        # set early stopping patience
        self.patience = 20
        self.patience_inc_coef = -0.1
        self.best_valid_result = np.inf

    def get_weighted_layers(self):
        weighted_layers = []
        for layer in self.layers:
            if layer.have_weights:
                weighted_layers.append(layer)
        return weighted_layers

    def connect(self):
        for i, layer in enumerate(self.layers[:-1]):
            self.layers[i+1].set_inputs_shape(layer.get_outputs_shape())

    def init_weights_baises(self):
        for layer in self.weighted_layers:
            layer.init_biases()
            layer.init_weights()

    def init_regularization(self):
        raise NotImplementedError

    def forward(self, x, batch_size=None):
        """
        :param x: array of inputs
        :param batch_size: can be ignored if not for conv layers etc.
        :return:
        """
        a = x
        for layer in self.layers:
            a = layer.forward(a, batch_size=batch_size)
        return a

    def train(self, data_sets, batch_size, n_epochs, learning_rate, cost_func, l1_a, l2_a):

        raise NotImplementedError

    def test(self, data_set):
        raise NotImplementedError

    def is_new_best(self, valid_result):
        """
        :param valid_result: errors found in validation. should always >=0 and <=1
        """
        assert 1 >= valid_result >= 0
        assert self.best_valid_result >= 0

        improvement_threshold = 0.005
        improvement = (valid_result - self.best_valid_result) / valid_result

        result = False
        if improvement > improvement_threshold:
            self.best_valid_result = valid_result
            self.patience += 3
            result = True
        else:
            self.patience *= 1 + self.patience_inc_coef
        print('improvement:{1}   patience left: {0}'.format(self.patience, improvement))

        if self.patience < 1:
            raise UserWarning('patience lost')

        return result

    def reset_params(self):
        self.init_weights_baises()
        self._set_weights_biases(*self._get_weights_biases())

    def save_params(self):
        weights, biases = self._get_weights_biases()
        with open(self.pickle_file, 'wb') as f:
            pickle.dump((weights, biases), f, protocol=pickle.HIGHEST_PROTOCOL)
        print('params saved.')

    def load_params(self):
        with open(self.pickle_file, 'rb') as f:
            params = pickle.load(f)
        # print(np_params[1])
        self._set_weights_biases(*params)
        print('params loaded.')

    def _get_weights_biases(self):
        raise NotImplementedError

    def _set_weights_biases(self, weights, biases):
        raise NotImplementedError

    def _get_b(self):
        raise NotImplementedError
