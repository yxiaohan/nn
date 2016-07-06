import os
import pickle
import sys

import base_layer
import conf
from my_tensor_flow.tf_layers import *
import my_tensor_flow.tf_ultilities as tfu
import common


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
    def init_cetptron_layers(layer_types: [NeuronLayer]):
        return [NeuronLayer(*layer_type) for layer_type in layer_types]

    @staticmethod
    def init_layers(inputs_shape, layer_types):
        input_layer = base_layer.DirectLayer(inputs_shape=inputs_shape)
        layers = [input_layer] + BaseNet.init_cetptron_layers(layer_types)
        return layers

    @classmethod
    def net_from_layer_types(cls, inputs_shape, layer_types):
        input_layer = base_layer.DirectLayer(inputs_shape=inputs_shape)
        layers = [input_layer] + cls.init_cetptron_layers(layer_types)
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
        # init tf session
        self.sess = tf.InteractiveSession()

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
        self.weights, self.biases = self._get_tf_weights_biases()

        # l1 and l2 regularization
        self.l1, self.l2 = self.init_regularization()

        # init tf variables
        self.x = tf.placeholder(tf.float32, shape=(None,) + self.layers[0].get_inputs_shape(), name='x')
        self.y = tf.placeholder(tf.float32, shape=(None,) + self.layers[-1].get_outputs_shape(), name='y')
        self.p_y = self.forward(self.x)

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
        l1 = 0
        l2 = 0
        for layer in self.weighted_layers:
            w = layer.w
            l1 += tf.reduce_sum(abs(w))
            l2 += tf.reduce_sum(w ** 2)
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
            a = layer.forward(a, batch_size=batch_size)
        return a

    def train(self, data_sets, batch_size, n_epochs, learning_rate=0.03,
              cost_func=tfu.Costs.cross_entropy, l1_a=0.0, l2_a=0.0001):

        cost = cost_func(self.p_y, self.y) + self.l1 * l1_a + self.l2 * l2_a

        # set early stopping patience
        self.patience = 20
        self.best_valid_result = 0

        # set up training step
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # train
        print('init all variables...')
        tf.initialize_all_variables().run()
        self._get_b()
        self.save_np_params()
        print('init finished, now start training...')
        train_set, valid_set, test_set = data_sets
        train_set_x, train_set_y = train_set
        n_train_batches = self.get_batch_number(train_set, batch_size)

        speed_test = common.SpeedTest()
        for epoch in range(n_epochs):
            print('n_epochs:%d/%d' % (epoch, n_epochs))
            speed_test.start()
            for batch_num in range(n_train_batches):
                mini_x = self._get_mini_batch(train_set_x, batch_size, batch_num)
                mini_y = self._get_mini_batch(train_set_y, batch_size, batch_num)
                train_step.run({self.x: mini_x, self.y: mini_y})
            speed_test.stop()
            valid_result = self.test(valid_set).item()
            print('validation result: %f' % valid_result)
            try:
                if self.is_new_best(valid_result):
                    print('new best found, saving...')
                    self.save_np_params()
            except UserWarning as e:
                print(e)
                return
        # print('test results: %f') % self.test(test_set)

    def test(self, data_set):
        set_x, set_y = data_set
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.p_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        model = accuracy.eval({self.x: set_x, self.y: set_y})
        return model

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
        self._set_weights_biases(*self._get_tf_weights_biases())

    def save_np_params(self):
        try:
            np_weights = self.sess.run(self.weights)
            np_biases = self.sess.run(self.biases)
            with open(self.pickle_file, 'wb') as f:
                pickle.dump((np_weights, np_biases), f, protocol=pickle.HIGHEST_PROTOCOL)
            print('params saved.')
        except tf.python.framework.errors.FailedPreconditionError:
            print('saving failed because not all variables are initiated.')

    def load_np_params(self):
        with open(self.pickle_file, 'rb') as f:
            np_params = pickle.load(f)
        tf_params = [[tf.Variable(p) for p in np_param] for np_param in np_params]
        # print(np_params[1])
        self._set_weights_biases(*tf_params)
        print('params loaded.')

    def _get_tf_weights_biases(self):
        tf_weights = []
        tf_biases = []
        for layer in self.weighted_layers:
            tf_weights.append(layer.w)
            tf_biases.append(layer.b)
        return tf_weights, tf_biases

    def _set_weights_biases(self, weights, biases):
        for i, layer in enumerate(self.weighted_layers):
            # to make save / load / reset functional
            tf.initialize_variables([weights[i], biases[i]]).run()
            self.sess.run(layer.w.assign(weights[i]))
            self.sess.run(layer.b.assign(biases[i]))

    def _get_b(self):
        common.print_func_name(self._get_b)
        layer = self.weighted_layers[0]
        print(self.sess.run(layer.b))

