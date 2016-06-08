import numpy as np
import math
import tensorflow as tf


class Ceptron(object):
    @staticmethod
    def core_func(z):
        raise NotImplementedError

    # def core_func_derivative(self, z):
    #     raise NotImplementedError
    @staticmethod
    def _init_weight_values(n_inputs, n_outputs, shape):
        return tf.zeros(shape)

    def init_weights(self, n_inputs, n_outputs, shape=None, scope=None):
        """
        this function should be rewrote by children classes
        default would return all zeros
        :param n_inputs: number of inputs
        :param n_outputs: number of outputs
        """
        with tf.name_scope(scope):
            if shape is None:
                shape = (n_inputs, n_outputs)
            w = tf.Variable(self._init_weight_values(n_inputs, n_outputs, shape), name='w')
        return w


class Sigmoid(Ceptron):
    @staticmethod
    def core_func(z):
        return tf.nn.sigmoid(z)

    @staticmethod
    def _init_weight_values(n_inputs, n_outputs, shape):
        distance = np.sqrt(6. / (n_inputs + n_outputs)) * 4
        w_values = tf.random_uniform(shape=shape, minval=-distance, maxval=distance)
        return w_values


class SoftMax(Ceptron):
    @staticmethod
    def core_func(z):
        return tf.nn.softmax(z)


class Tanh(Ceptron):
    @staticmethod
    def core_func(z):
        return tf.nn.tanh(z)

    @staticmethod
    def _init_weight_values(n_inputs, n_outputs, shape):
        distance = np.sqrt(6. / (n_inputs + n_outputs))
        w_values = tf.random_uniform(shape=shape, minval=-distance, maxval=distance)
        return w_values


class Relu(Ceptron):
    @staticmethod
    def core_func(z):
        return tf.nn.relu(z)

    @staticmethod
    def _init_weight_values(n_inputs, n_outputs, shape):
        w_values = tf.truncated_normal(shape=shape, stddev=1. / math.sqrt(float(shape[0])))
        return w_values

