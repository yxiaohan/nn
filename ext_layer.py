from theano.tensor.signal import pool

from base_layer import CeptronLayer, AbstractLayer
import ceptron


class DirectLayer(AbstractLayer):
    """
    this layer simply bypasses inputs to outputs, without any changes, usually acting as the first layer
    """
    def __init__(self, inputs_shape: tuple):
        super().__init__(outputs_shape=inputs_shape)

    def forward(self, inputs):
        return inputs


class PoolingLayer(AbstractLayer):
    """
    layer for pooling convolutional outputs
    """
    def __init__(self, inputs_shape: tuple, pool_size: tuple):
        """
        :type inputs_shape: a tuple
        :param inputs_shape: the shape of inputs
        """
        assert len(inputs_shape) == len(pool_size)
        outputs_shape = tuple([i // p for i, p in zip(inputs_shape, pool_size)])
        self.pool_size = pool_size
        super().__init__(outputs_shape=outputs_shape)

    def forward(self, inputs):
        raise NotImplementedError


class MaxPoolingLayer(PoolingLayer):
    def forward(self, inputs):
        return pool.max_pool_2d(input=inputs, ds=self.pool_size, ignore_border=True)


class ConvLayer(CeptronLayer):
    """
    layer for doing convolutional job
    """
    def __init__(self, filter_shape, ceptron=ceptron.Tanh):
        pow()