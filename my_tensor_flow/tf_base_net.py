import base_layer
from my_tensor_flow.tf_layers import *
import my_tensor_flow.tf_ultilities as tfu
import common
import base_net


class BaseNet(base_net.BaseNet):
    @staticmethod
    def init_neuron_layers(layer_types: [NeuronLayer]):
        return [NeuronLayer(*layer_type) for layer_type in layer_types]

    @staticmethod
    def _np_params_to_tf_params(np_params):
        return [tf.Variable(p) for p in np_params]

    def __init__(self, layers: [base_layer.AbstractLayer]):
        """
        :parameter layers: a list of tuple for init layers
        """
        # init tf session
        self.sess = tf.InteractiveSession()

        super().__init__(layers)

        # self.weights, self.biases = self._get_tf_weights_biases()

        # init tf variables
        self.x = tf.placeholder(tf.float32, shape=(None,) + self.layers[0].get_inputs_shape(), name='x')
        self.y = tf.placeholder(tf.float32, shape=(None,) + self.layers[-1].get_outputs_shape(), name='y')
        self.p_y = self.forward(self.x)

    def init_regularization(self):
        l1 = 0
        l2 = 0
        for layer in self.weighted_layers:
            w = layer.w
            l1 += tf.reduce_sum(abs(w))
            l2 += tf.reduce_sum(w ** 2)
        # l2 = T.sqrt(l2)
        return l1, l2

    def train(self, data_sets, batch_size, n_epochs,
              learning_rate=0.03, cost_func=tfu.Costs.cross_entropy, l1_a=0.0, l2_a=0.0001):

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
        self.save_params()
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
                    self.save_params()
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

    def reset_params(self):
        self.init_weights_baises()
        self._set_weights_biases(*self._get_tf_weights_biases())

    # def save_np_params(self):
    #     try:
    #         np_weights = self.sess.run(self.weights)
    #         np_biases = self.sess.run(self.biases)
    #         with open(self.pickle_file, 'wb') as f:
    #             pickle.dump((np_weights, np_biases), f, protocol=pickle.HIGHEST_PROTOCOL)
    #         print('params saved.')
    #     except tf.python.framework.errors.FailedPreconditionError:
    #         print('saving failed because not all variables are initiated.')

    # def load_np_params(self):
    #     with open(self.pickle_file, 'rb') as f:
    #         np_params = pickle.load(f)
    #     tf_params = [[tf.Variable(p) for p in np_param] for np_param in np_params]
    #     # print(np_params[1])
    #     self._set_weights_biases(*tf_params)
    #     print('params loaded.')

    def _tf_params_to_np_params(self, tf_params):
        try:
            return self.sess.run(tf_params)
        except tf.python.framework.errors.FailedPreconditionError:
            print('saving failed because not all variables are initiated.')

    def _get_weights_biases(self):
        tf_weights, tf_biases = self._get_tf_weights_biases()
        weights = self._tf_params_to_np_params(tf_weights)
        biases = self._tf_params_to_np_params(tf_biases)
        return weights, biases

    # return params in tensorflow types
    def _get_tf_weights_biases(self):
        tf_weights = []
        tf_biases = []
        for layer in self.weighted_layers:
            tf_weights.append(layer.w)
            tf_biases.append(layer.b)
        return tf_weights, tf_biases

    def _set_weights_biases(self, weights, biases):
        tf_weights = self._np_params_to_tf_params(weights)
        tf_biases = self._np_params_to_tf_params(biases)
        for i, layer in enumerate(self.weighted_layers):
            # to make save / load / reset functional
            tf.initialize_variables([tf_weights[i], tf_biases[i]]).run()
            self.sess.run(layer.w.assign(tf_weights[i]))
            self.sess.run(layer.b.assign(tf_biases[i]))

    def _get_b(self):
        common.print_func_name(self._get_b)
        layer = self.weighted_layers[0]
        print(self.sess.run(layer.b))
