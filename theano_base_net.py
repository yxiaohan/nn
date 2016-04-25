import theano
import theano.tensor as T
import numpy as np
import pickle
import sys
import os

import conf
import base_layer
import ext_layer
import theano_utilities as tu


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
        self.weighted_layers = self.get_weighted_layers()
        self.rng = np.random.RandomState(1234)
        self.depth = len(self.weighted_layers)

        # set pickle file path
        self.file_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.pickle_file = conf.DATA_PATH + self.file_name + '.pkl'

        self.connect()
        # self.reset_params()
        self.init_weights_baises()
        self.weights, self.biases = self._get_weights_biases()

        # l1 and l2 regularization
        self.l1, self.l2 = self.init_regularization()

        # init theano functions
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.index = T.iscalar('index')

        self.train_model = None
        self.valid_model = None
        self.test_model = None

        # set early stopping patience
        self.patience = 20
        self.patience_inc_coef = -0.1
        self.lest_valid_error = np.inf

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
            layer.init_weights(self.rng)

    def init_regularization(self):
        l1 = 0
        l2 = 0
        for layer in self.weighted_layers:
            w = layer.w
            l1 += T.sum(abs(w))
            l2 += T.sum(w ** 2)
        # l2 = T.sqrt(l2)
        return l1, l2

    def forward(self, x, batch_size=None):
        """
        :param x: array of inputs
        :param batch_size: can be ignored if not for conv layers etc.
        :return:
        """
        a = x
        for layer in self.layers:
            a = layer.forward(a, {'batch_size': batch_size})
        return a

    def set_train_model(self, train_set, cost_func, batch_size, learning_rate, l1_a=0.0, l2_a=0.0001):
        self.p_y = self.forward(self.x)
        cost = cost_func(self.p_y, self.y) + self.l1 * l1_a + self.l2 * l2_a

        # set early stopping patience
        self.patience = 20
        self.lest_valid_error = np.inf

        print('compiling train model..')

        # compute gradients of weights and biases
        updates = []
        for layer in reversed(self.weighted_layers):
            g_w = T.grad(cost, layer.w)
            g_b = T.grad(cost, layer.b)
            updates += [(layer.w, layer.w - learning_rate * g_w),
                        (layer.b, layer.b - learning_rate * g_b)]

        train_set_x, train_set_y = train_set
        index = self.index
        self.train_model = theano.function([index], cost, updates=updates, givens={
            self.x: self._get_mini_batch(train_set_x, batch_size, index),
            self.y: self._get_mini_batch(train_set_y, batch_size, index)
        })

        # check if using gpu
        tu.check_gpu(self.train_model)

    def init_with_train_model(self, layers, train_set, cost_func, batch_size, learning_rate, l1_a=0.0, l2_a=0.0001):
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
        self.weights, self.biases = self._get_weights_biases()

        # l1 and l2 regularization
        self.l1, self.l2 = self.init_regularization()

        # init theano functions
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.index = T.iscalar('index')

        self.train_model = None
        self.valid_model = None
        self.test_model = None

        # set early stopping patience
        self.patience = 20
        self.patience_inc_coef = -0.1
        self.lest_valid_error = np.inf
        self.set_train_model(train_set, cost_func, batch_size, learning_rate, l1_a, l2_a)

    def set_valid_test_models(self, valid_set, test_set, errors):
        print('setting valid and test models...')
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

        improvement_threshold = 0.05
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
        self.init_weights_baises()
        self._set_weights_biases(*self._get_weights_biases())

    def save_params(self):
        weights, biases = self._get_weights_biases()
        with open(self.pickle_file, 'wb') as f:
            pickle.dump((weights, biases), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_params(self):
        with open(self.pickle_file, 'rb') as f:
            params = pickle.load(f)
        self._set_weights_biases(*params)

    def _get_weights_biases(self):
        weights = []
        biases = []
        for layer in self.weighted_layers:
            weights.append(layer.w)
            biases.append(layer.b)
        return weights, biases

    def _set_weights_biases(self, weights, biases):
        for i, layer in enumerate(self.weighted_layers):
            layer.w.set_value(weights[i].get_value())
            layer.b.set_value(biases[i].get_value())
            # to make save / load / reset functional
            self.weights[i].set_value(weights[i].get_value())
            self.biases[i].set_value(biases[i].get_value())

    def get_b(self):
        print('get_b:')
        layer = self.weighted_layers[1]
        print(layer.b.get_value())
        # print(layer.w.get_value())
        _get_b = theano.function([], layer.b)
        print(_get_b())

