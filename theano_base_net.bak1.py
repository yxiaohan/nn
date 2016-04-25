import theano
import theano.tensor as T
import numpy as np
import pickle
import sys
import os

import conf
import my_ceptron
import base_layer
import ext_layer
import theano_utilities as tu
import theano_cost_function


class BaseNet(object):
    @staticmethod
    def list_of_same_length(l):
        return [0] * len(l)

    @staticmethod
    def _get_mini_batch(data_set, batch_size, batch_index):
        return data_set[batch_index * batch_size: (batch_index + 1) * batch_size]

    @staticmethod
    def get_batch_number(data_set, batch_size):
        return data_set[0].get_value(borrow=True).shape[0] // batch_size

    @staticmethod
    def init_cetptron_layers(layer_types):
        return [base_layer.CeptronLayer(*layer_type) for layer_type in layer_types]

    @classmethod
    def net_from_layer_types(cls, inputs_shape, layer_types):
        input_layer = ext_layer.DirectLayer(inputs_shape=inputs_shape)
        layers = [input_layer] + cls.init_cetptron_layers(layer_types)
        net = cls(layers)
        return net

    def __init__(self, layers: [base_layer.AbstractLayer]):
        """
        :parameter layers: a list of tuple for init layers
        """
        self.layers = layers
        self.rng = np.random.RandomState(1234)
        self.depth = len(self.layers) - 1

        self.weights = self.init_weights()
        self.biases = self.init_biases()

        # l1 and l2 regularization
        self.l1, self.l2 = self.init_regularization()

        # init theano functions
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.index = T.iscalar('index')
        self.p_y = self.forward(self.x)

        self.train_model = None
        self.valid_model = None
        self.test_model = None
        # self.set_weights_biases = theano.function(self._make_updates(self.weights, self.biases))

        # set early stopping patience
        self.patience = 20
        self.patience_inc_coef = -0.1
        self.lest_valid_error = np.inf

        # set pickle file path
        self.file_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.pickle_file = conf.DATA_PATH + self.file_name + '.pkl'

    def init_weights(self):
        weights = []
        for i in range(self.depth):
            w = self.layers[i + 1].init_weights(
                self.rng, self.layers[i].n_outputs, self.layers[i+1].n_outputs)
            self.layers[i + 1].set_weights(w)
            weights.append(w)
        return weights

    def init_biases(self):
        return [layer.init_biases() for layer in self.layers[1:]]

    def init_regularization(self):
        l1 = 0
        l2 = 0
        for w in self.weights:
            l1 += T.sum(abs(w))
            l2 += T.sum(w ** 2)
        # l2 = T.sqrt(l2)
        return l1, l2

    def forward(self, x):
        """
        :param x: array of inputs
        :return:
        """
        a = x
        for w, b, layer in zip(self.weights, self.biases, self.layers[1:]):
            z = T.dot(a, w) + b
            a = layer.ceptron.core_func(z=z)
        return a

    def set_train_model(self, train_set, cost_func, batch_size, learning_rate, l1_a=0.0, l2_a=0.0001):

        cost = cost_func(self.p_y, self.y) \
               + self.l1 * l1_a + self.l2 * l2_a

        print('compiling train model..')

        # compute gradients of weights and biases
        updates = []
        for i in range(self.depth):
            g_w = T.grad(cost, self.weights[i])
            g_b = T.grad(cost, self.biases[i])
            updates += [(self.weights[i], self.weights[i] - learning_rate * g_w),
                        (self.biases[i], self.biases[i] - learning_rate * g_b)]

        # print(self.train_set[1:3])
        train_set_x, train_set_y = train_set
        index = self.index
        self.train_model = theano.function([index], cost, updates=updates, givens={
            self.x: self._get_mini_batch(train_set_x, batch_size, index),
            self.y: self._get_mini_batch(train_set_y, batch_size, index)
        })

        # check if using gpu
        tu.check_gpu(self.train_model)

    def set_valid_test_models(self, valid_set, test_set, errors):
        self.valid_model = self._set_model(valid_set, errors)
        self.test_model = self._set_model(test_set, errors)

    def _set_model(self, data_set, errors, batch_size=None):
        set_x, set_y = data_set
        if batch_size is None:
            model = theano.function([], errors, givens={
                self.x: set_x,
                self.y: set_y
            })
        else:
            raise NotImplementedError
        return model

    def is_new_best(self, valid_error):
        """
        :param valid_error: errors found in validation. should always >=0
        """
        assert valid_error >= 0
        assert self.lest_valid_error >= 0

        improvement_threshold = 0.0005
        improvement = (self.lest_valid_error - valid_error) / valid_error

        result = False
        if improvement > improvement_threshold:
            self.lest_valid_error = valid_error
            self.patience += 3
            result = True
        else:
            self.patience *= 1 + self.patience_inc_coef
        # print('improvement:{1}   patience left: {0}'.format(self.patience, improvement))

        if self.patience < 1:
            raise UserWarning('patience lost')

        return result

    def reset_params(self):
        self._set_weights_biases(self.init_weights(), self.init_biases())

    def save_params(self):
        with open(self.pickle_file, 'wb') as f:
            pickle.dump((self.weights, self.biases), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self):
        with open(self.pickle_file, 'rb') as f:
            params = pickle.load(f)
        # _load_params = self._set_weights_biases(*params)
        # _load_params()
        self._set_weights_biases(*params)

    def _set_weights_biases(self, weights, biases):
        for i in range(self.depth):
            self.weights[i].set_value(weights[i].get_value())
            self.biases[i].set_value(biases[i].get_value())

    def get_b(self):
        print(self.biases[0].get_value())
        print(self.weights[0].get_value())
        _get_b = theano.function([], self.biases[0])
        print(_get_b())

    def _make_updates(self, weights, biases):
        updates = []
        for i in range(self.depth):
            updates += [(self.weights[i], weights[i]),
                        (self.biases[i], biases[i])]
        return updates
