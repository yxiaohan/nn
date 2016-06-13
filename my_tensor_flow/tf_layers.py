from base_layer import *
from my_tensor_flow.tf_neurons import *


class NeuronLayer(AbstractLayer):
    """
    this layer expects a neuron
    """
    def __init__(self, outputs_shape, neuron, w=None, b=None, scope=None):
        """
        :argument neuron: type of core_functions etc.
        :argument scope: scope name used for layer in tensor flow
        """
        super().__init__(outputs_shape)
        self.neuron = neuron
        if not isinstance(neuron, Neuron):
            print(neuron)
            raise ValueError('neuron must be a type of my_neuron.Neuron')
        self.have_weights = True
        self.b = b
        self.w = w
        self.scope = scope

    def __repr__(self):
        return 'layer({!r}, {!r})'.format(type(self.neuron), self.get_outputs_shape())

    def connect(self, layer_to_connect, layer_index=None):
        super().connect(layer_to_connect)
        # set name scope
        if self.scope is None:
            self.scope = 'layer_%d' % layer_index

    def forward(self, inputs, **kwargs):
        # inputs = inputs.flatten(2)
        print(self.w.get_shape())
        z = tf.matmul(inputs, self.w) + self.b
        return self.neuron.core_func(z)

    def init_weights(self):
        n_inputs = np.prod(self.get_inputs_shape())
        n_outputs = np.prod(self.get_outputs_shape())

        self.w = self.neuron.init_weights(n_inputs, n_outputs, scope=self.scope)

    def init_biases(self):
        with tf.name_scope(self.scope):
            self.b = tf.Variable(tf.zeros(self.get_outputs_shape()), name='b')


class MaxPoolingLayer(PoolingLayer):
    def forward(self, inputs, **args):
        ksize = [1] + list(self.pool_size) + [1]
        return tf.nn.max_pool(inputs, ksize=ksize, strides=ksize, padding='VALID')


class Conv2DLayer(NeuronLayer):
    """
    layer for doing convolutional job, handling 2d arrays
    """
    def __init__(self, n_feature_map, filter_shape, neuron=Tanh()):
        """
        :param filter_shape: a length 2 tuple with height and width of filter
        :param n_feature_map: number of feature maps
        """
        outputs_shape = filter_shape + (n_feature_map, )
        super().__init__(outputs_shape, neuron)
        self.filter_shape = filter_shape
        self.n_feature_map = n_feature_map

    def __repr__(self):
        return 'layer({!r}, {!r})'.format(type(self), self.get_outputs_shape())

    def get_outputs_shape(self):
        if self._inputs_shape is None:
            raise ValueError("output_shape is not correct when inputs_shape is not defined yet.")
        return super().get_outputs_shape()

    def set_inputs_shape(self, inputs_shape: tuple):
        """
        in convlayers, the outputs_shape and inputs_shape are linked, thus will be set at same time
        transfer the common inputs_shape (height), width), n_input_maps) into
        a length 3 tuple(n_input_maps, height, width)
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
        print(self._inputs_shape, self.filter_shape)
        # the shape of outputs should be (length, width, n_feature_map)
        self.set_outputs_shape((height - self.filter_shape[0] + 1,
                                width - self.filter_shape[1] + 1, self.n_feature_map))

    def init_weights(self):
        n_filter_cells = np.prod(self.filter_shape)
        n_inputs = self.get_inputs_shape()[0] * n_filter_cells
        n_outputs = self.n_feature_map * n_filter_cells
        shape = (self.n_feature_map, self._inputs_shape[0]) + self.filter_shape
        self.w = self.neuron.init_weights(n_inputs, n_outputs, shape)

    def init_biases(self):
        self.b = tf.Variable(tf.zeros(self.n_feature_map), name='b')

    def convolution(self, inputs):
        # print(inputs.shape)
        # batch_size = inputs.shape[0]
        # convert inputs into con2d required shapes
        # n_input_maps, height, width = self._inputs_shape
        # x = inputs.reshape(batch_size, n_input_maps, height, width)
        input_shape = (-1, ) + super().get_inputs_shape()
        print(input_shape)
        inputs = inputs.reshape(input_shape)
        # filter_shape_4d = (self.n_feature_map, self._inputs_shape[0]) + self.filter_shape
        z = tf.nn.conv2d(input=inputs, filter=self.w, strides=[1, 1, 1, 1], padding='VALID')

        return z

    def pre_forward(self, inputs):
        z = self.convolution(inputs)
        return z

    def forward(self, inputs, **kwargs):
        z = self.pre_forward(inputs)
        z = tf.nn.relu(z + self.b)
        # z = z.flatten(2)
        # reshape to 2d
        return z


class Conv2DPoolingLayer(Conv2DLayer):
    def __init__(self, n_feature_map, filter_shape, pool_size, neuron=Tanh()):
        super().__init__(n_feature_map, filter_shape, neuron)
        self.pool_size = pool_size
        # pooling layer will be initiated after inputs shape is settled
        self.pooling_layer = None

    def set_inputs_shape(self, inputs_shape: tuple):
        super().set_inputs_shape(inputs_shape)
        pool_size = self.pool_size
        outputs_shape = self.get_outputs_shape()
        self.pooling_layer = MaxPoolingLayer(outputs_shape[:len(pool_size)], pool_size)
        self.set_outputs_shape(self.pooling_layer.get_outputs_shape() + outputs_shape[len(pool_size):])

    def pre_forward(self, inputs):
        z = super().pre_forward(inputs)
        z = self.pooling_layer.forward(z)
        return z
