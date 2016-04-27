from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
import theano.tensor as T
import numpy as np

from base_layer import CeptronLayer, AbstractLayer
import theano_utilities as tu
import my_ceptron


class DirectLayer(AbstractLayer):
    """
    this layer simply bypasses inputs to outputs, without any changes, usually acting as the first layer
    """
    def __init__(self, inputs_shape: tuple):
        super().__init__(outputs_shape=inputs_shape)

    def forward(self, inputs, **args):
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

    def forward(self, inputs, **args):
        raise NotImplementedError


class MaxPoolingLayer(PoolingLayer):
    def forward(self, inputs, **args):
        return downsample.max_pool_2d(input=inputs, ds=self.pool_size, ignore_border=True)


class Conv2DLayer(CeptronLayer):
    """
    layer for doing convolutional job, handling 2d arrays
    """
    def __init__(self, n_feature_map, filter_shape, ceptron=my_ceptron.Tanh()):
        """
        :param filter_shape: a length 2 tuple with height and width of filter
        :param n_feature_map: number of feature maps
        """
        outputs_shape = filter_shape + (n_feature_map, )
        print(ceptron)
        super().__init__(outputs_shape, ceptron)
        self.filter_shape = filter_shape
        self.n_feature_map = n_feature_map

    def __repr__(self):
        return 'layer({!r}, {!r})'.format(type(self), self._outputs_shape)

    def get_outputs_shape(self):
        if self._inputs_shape is None:
            raise ValueError("output_shape is not correct when inputs_shape is not defined yet.")
        return self._outputs_shape

    def set_inputs_shape(self, inputs_shape: tuple):
        """
        in convlayers, the outputs_shape and inputs_shape are linked, thus will be set at same time
        transfer the common inputs_shape (height), width), n_input_maps) into a length 3 tuple(n_input_maps, height, width)
        """
        if len(inputs_shape) == 1:
            self._inputs_shape = (1, inputs_shape[0], 1)
        elif len(inputs_shape) == 2:
            self._inputs_shape = (1, ) + inputs_shape
        elif len(inputs_shape) == 3:
            self._inputs_shape = (inputs_shape[2], inputs_shape[0], inputs_shape[1])
        else:
            raise TypeError('inputs_shape length should be 1-3 (height), width), n_input_maps)')

        n_input_maps, height, width = self._inputs_shape

        # the shape of outputs should be (length, width, n_feature_map)
        self._outputs_shape = (height - self.filter_shape[0] + 1, width - self.filter_shape[1] + 1, self.n_feature_map)

    def init_weights(self, rng):
        n_filter_cells = np.prod(self.filter_shape)
        n_inputs = self.get_inputs_shape()[0] * n_filter_cells
        n_outputs = self.n_feature_map * n_filter_cells
        shape = (self.n_feature_map, self._inputs_shape[0]) + self.filter_shape
        self.w = self.ceptron.weights_init_func(rng, n_inputs, n_outputs, shape)

    def init_biases(self):
        self.b = tu.shared_zeros(self.n_feature_map, 'b')

    def convolution(self, inputs, batch_size):
        # print(inputs.shape)
        # batch_size = inputs.shape[0]
        # convert inputs into con2d required shapes
        # n_input_maps, height, width = self._inputs_shape
        # x = inputs.reshape(batch_size, n_input_maps, height, width)
        input_shape = (batch_size, ) + self._inputs_shape
        print(input_shape)
        inputs = inputs.reshape(input_shape)
        filter_shape_4d = (self.n_feature_map, self._inputs_shape[0]) + self.filter_shape
        z = conv2d(input=inputs, filters=self.w, input_shape=input_shape, filter_shape=filter_shape_4d)
        return z

    def pre_forward(self, inputs, batch_size):
        z = self.convolution(inputs, batch_size)
        return z

    def forward(self, inputs, **kwargs):
        key = 'batch_size'
        if key in kwargs:
            batch_size = kwargs[key]
        else:
            raise ValueError('batch_size not specified')
        z = self.pre_forward(inputs, batch_size)
        # z = super().forward(z)
        z = T.tanh(z + self.b.dimshuffle('x', 0, 'x', 'x'))
        z = z.flatten(2)
        return z


class Conv2DPoolingLayer(Conv2DLayer):
    def __init__(self, n_feature_map, filter_shape, pool_size, ceptron=my_ceptron.Tanh):
        super().__init__(n_feature_map, filter_shape, ceptron)
        self.pooling_layer = MaxPoolingLayer(filter_shape, pool_size)

    def pre_forward(self, inputs, batch_size):
        z = super().pre_forward(inputs, batch_size)
        z = self.pooling_layer.forward(z)
        return z
