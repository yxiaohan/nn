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
            data_shape = (data_shape, )
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

        self.set_outputs_shape(AbstractLayer._set_data_shape(outputs_shape))
        # inputs shape will be set after the layer is connected in a network
        self._inputs_shape = None

    def __repr__(self):
        return 'layer({!r}, {!r})'.format(type(self), self.get_outputs_shape())

    def set_inputs_shape(self, inputs_shape):
        self._inputs_shape = AbstractLayer._set_data_shape(inputs_shape)

    def get_inputs_shape(self):
        return self._inputs_shape

    def get_outputs_shape(self):
        return self._outputs_shape

    def set_outputs_shape(self, outputs_shape):
        self._outputs_shape = outputs_shape

    def connect(self, layer_to_connect, layer_index=None):
        """
        set up out puts shape, based on the layer connected to
        :param layer_to_connect:
        :param layer_index: the index of this layer (self) in network
        :return:
        """
        assert isinstance(layer_to_connect, AbstractLayer)
        self.set_inputs_shape(layer_to_connect.get_outputs_shape())

    def forward(self, inputs, **kwargs):
        raise NotImplementedError


class DirectLayer(AbstractLayer):
    """
    this layer simply bypasses inputs to outputs, without any changes, usually acting as the first layer
    """
    def __init__(self, inputs_shape: tuple):
        super().__init__(outputs_shape=inputs_shape)
        self.set_inputs_shape(self.get_outputs_shape())

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
        print(inputs_shape, pool_size)
        assert len(inputs_shape) == len(pool_size)
        outputs_shape = tuple([i // p for i, p in zip(inputs_shape, pool_size)])
        self.pool_size = pool_size
        super().__init__(outputs_shape=outputs_shape)

    def forward(self, inputs, **args):
        raise NotImplementedError
