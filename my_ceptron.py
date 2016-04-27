import numpy as np
import theano.tensor as T
import theano

import theano_utilities as tu


class Ceptron(object):
    @staticmethod
    def core_func(z):
        raise NotImplementedError

    # def core_func_derivative(self, z):
    #     raise NotImplementedError

    def weights_init_func(self, rng, n_inputs, n_outputs):
        """
        this function should be rewrote by children classes
        default would return all zeros
        :param rng: passed random state to generate random values
        :param n_inputs: number of inputs
        :param n_outputs: number of outputs
        """
        return tu.shared_zeros((n_inputs, n_outputs), name='W')


class Segmoid(Ceptron):
    @staticmethod
    def core_func(z):
        return 1.0/(1.0+np.exp(-z))

    def core_func_derivative(self, z):
        return self.core_func(z)*(1-self.core_func(z))

    def weights_init_func(self, rng, n_inputs, n_outputs, shape=None):
        if shape is None:
            shape = (n_inputs, n_outputs)
        w_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_inputs + n_outputs)),
                high=np.sqrt(6. / (n_inputs + n_outputs)),
                size=shape
            ),
            dtype=theano.config.floatX
        )

        w_values *= 4

        w = theano.shared(value=w_values, name='W', borrow=True)
        return w


class TheanoSoftMax(Ceptron):
    @staticmethod
    def core_func(z):
        return T.nnet.softmax(z)


class Tanh(Ceptron):
    @staticmethod
    def core_func(z):
        return T.tanh(z)

    def weights_init_func(self, rng, n_inputs, n_outputs, shape=None):
        """
        :param rng: passed random state to generate random values
        :param n_inputs: number of inputs
        """
        if shape is None:
            shape = (n_inputs, n_outputs)
        w_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / (n_inputs + n_outputs)),
                high=np.sqrt(6. / (n_inputs + n_outputs)),
                size=shape
            ),
            dtype=theano.config.floatX
        )

        w = theano.shared(value=w_values, name='W', borrow=True)
        return w
