import theano.tensor as T
import numpy as np

import theano_utilities as tu
from ceptron import Ceptron


class AbstractLayer(object):
    """
    abstract layer without weights or biases, or ceptron
    """
    def __init__(self, outputs_shape):
        """
        :type outputs_shape: int or tuple
        :argument outputs_shape: shape of ceptrons / outputs
        """
        if type(outputs_shape) is int:
            outputs_shape = (outputs_shape,)
        elif type(outputs_shape) is tuple:
            pass
        else:
            raise ValueError('the outputs_shape must be a int or tuple!')
        n_outputs = np.prod(outputs_shape)
        self.n_outputs = n_outputs
        self.outputs_shape = outputs_shape

    def __repr__(self):
        return 'layer({!r}, {!r})'.format(type(self), self.outputs_shape)

    def forward(self, inputs):
        raise NotImplementedError


class CeptronLayer(AbstractLayer):
    """
    this layer expects a ceptron
    """
    def __init__(self, outputs_shape, ceptron, w=None, b=None):
        """
        :argument ceptron: type of core_functions etc.
        """
        super().__init__(outputs_shape)
        self.ceptron = ceptron
        if not isinstance(ceptron, Ceptron):
            print(type(ceptron))
            raise ValueError('ceptron must be a type of ceptron.Ceptron')
        self.b = b
        self.w = w

    def __repr__(self):
        return 'layer({!r}, {!r})'.format(type(self.ceptron), self.outputs_shape)

    def forward(self, inputs):
        z = T.dot(inputs, self.w) + self.b
        return self.ceptron.core_func(z)

    def init_weights(self, *args):
        return self.ceptron.weights_init_func(*args)

    def init_biases(self):
        return tu.shared_zeros(self.outputs_shape, 'b')

    def set_weights(self, w):
        self.w = w

    def set_biases(self, b):
        self.b = b

    def get_weights(self):
        return self.w

    def get_biases(self):
        return self.b
