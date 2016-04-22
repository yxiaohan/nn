import theano.tensor as T
import numpy as np

import theano_utilities as tu
from ceptron import Ceptron


class AbstractLayer(object):
    """
    abstract layer without weights or biases, or ceptron
    """
    @staticmethod
    def _set_data_shape(data_shape):
        """
        set the inputs and outputs shapes
        """
        if type(data_shape) is int:
            data_shape = (data_shape,)
        elif type(data_shape) is tuple:
            pass
        else:
            raise ValueError('the data_shape must be a int or tuple!')

        return data_shape

    def __init__(self, outputs_shape):
        """
        :type outputs_shape: int or tuple
        :argument outputs_shape: shape of ceptrons / outputs
        """
        self.have_weights = False

        self.outputs_shape = AbstractLayer._set_data_shape(outputs_shape)
        # inputs shape will be set after the layer is connected in a network
        self.inputs_shape = None

    def __repr__(self):
        return 'layer({!r}, {!r})'.format(type(self), self.outputs_shape)

    def set_inputs_shape(self, inputs_shape):
        self.inputs_shape = AbstractLayer._set_data_shape(inputs_shape)

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
        self.have_weights = True
        self.b = b
        self.w = w

    def __repr__(self):
        return 'layer({!r}, {!r})'.format(type(self.ceptron), self.outputs_shape)

    def forward(self, inputs):
        z = T.dot(inputs, self.w) + self.b
        return self.ceptron.core_func(z)

    def init_weights(self, rng):
        n_inputs = np.prod(self.inputs_shape)
        n_outputs = np.prod(self.outputs_shape)
        self.w = self.ceptron.weights_init_func(rng, n_inputs, n_outputs)

    def init_biases(self):
        self.b = tu.shared_zeros(self.outputs_shape, 'b')

    # def set_weights(self, w):
    #     self.w = w
    #
    # def set_biases(self, b):
    #     self.b = b
    #
    # def get_weights(self):
    #     return self.w
    #
    # def get_biases(self):
    #     return self.b
