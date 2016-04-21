import numpy as np
import mnist, common, ceptron, random
import matplotlib.pyplot as plt

# mn = mnist.MNIST()
class Networks(object):
    @staticmethod
    # returns a same shape of input array, filled with zeros
    def zeros_for_array(prototype_array):
        return np.zeros(prototype_array.shape)

    @staticmethod
    def add_one_d(arr):
        return arr.reshape(arr.shape + (1,))

    def init_list_of_np_arrays(self, l = None):
        if l is None:
            l = self.l
        return [np.zeros(1)] * l

    def __init__(self, layers, ceptron=None, weight_init_func=None, layer_init_func=np.random.randn):
        self.layers = layers
        self.ceptron = ceptron
        self.l = len(layers) - 1
        # init weights and bias, where weights have the shape of (layer[i+1], layer[i])
        self.weights = self.init_list_of_np_arrays()
        self.biases = self.init_list_of_np_arrays()
        if weight_init_func is None:
            weight_init_func = layer_init_func
        for i in range(self.l):
            self.weights[i] = weight_init_func(layers[i+1], layers[i])
            self.biases[i] = layer_init_func(layers[i + 1], 1)

    # a means the outputs of last layer
    def cost(self, y, a=None):
        if a is None:
            a = self.outputs[-1]
        return ((y - a)**2).mean()/2

    def cost_derivative(self, y, a):
        # y.reshape(a.shape)
        # print(y.shape, a.shape)
        return y - a

    def forward(self, inputs, w, b):
        # print(inputs.shape, w.shape, b.shape)
        # print(inputs.shape)
        z = w.dot(inputs) + b
        return self.ceptron.core_func(z), z

    def compute(self, x):
        # print(x.shape, self.layers[0])
        # if x.shape[-1] != self.layers[0]:
        #     raise NotImplementedError

        outputs = [x] # out puts of first layer
        zs = []
        for w, b in zip(self.weights, self.biases):
            output, z = self.forward(outputs[-1], w, b)
            outputs.append(output)
            zs.append(z)
        return outputs, zs

    # adjust weights and biases based on given training x and y
    def back_prop(self, x, y):
        outputs, zs = self.compute(x)
        delta_weights = [0] * self.l
        delta_biases = self.init_list_of_np_arrays()
        errors = self.init_list_of_np_arrays()

        # compute errors backwards
        for i in range(self.l - 1, -1, -1):
            # print(self.zs[i].shape, i)
            if i == self.l - 1: # last layer
                errors[i] = self.cost_derivative(y, outputs[-1])\
                             * self.ceptron.core_func_derivative(zs[i])
            else:
                errors[i] = errors[i + 1].dot(self.weights[i+1].T) \
                                * self.ceptron.core_func_derivative(zs[i])

            delta_biases[i] = errors[i].sum(axis=0)
            pre_outputs = outputs[i]
            # print(errors[i].shape, pre_outputs.shape)
            for m in range(pre_outputs.shape[0]):
                o = pre_outputs[m]
                e = errors[i][m]
                et = e.reshape(e.shape + (1,)).T
                # print(o.shape, et.shape)
                delta_weights[i] += o.reshape(o.shape + (1,)).dot(et)
            # print(delta_weights[i].shape)
        return delta_weights, delta_biases

    def back_prop1(self, x, y):
        outputs, zs = self.compute(x)
        delta_weights = [0] * self.l
        delta_biases = self.init_list_of_np_arrays()

        # compute errors backwards
        errors = None
        a = outputs[-1]
        for i in range(self.l - 1, -1, -1):
            # print(y.shape, zs[i].shape)
            # print(self.cost_derivative(y, a).shape)
            if i == self.l - 1: # last layer
                errors = self.cost_derivative(y, a)\
                             * self.ceptron.core_func_derivative(zs[i])
            else:
                errors = delta_weights[i+1].T.dot(errors) \
                                * self.ceptron.core_func_derivative(zs[i])
            delta_biases[i] = errors
            pre_outputs = outputs[i]
            # print(errors[i].shape, pre_outputs.shape)
            delta_weights[i] = (errors).dot((pre_outputs).T)
            # print('delta_weights:' + str(delta_weights[i].shape))
            # print('delta_biases:' + str(delta_biases[i].shape))
        return delta_weights, delta_biases

    def mini_batch(self, batch_size, x_set, y_set, eta=3):
        train_len = len(x_set)
        if len(y_set) != train_len:
            raise ValueError('lengths of x, y are not same')
        if batch_size >= train_len:
            raise ValueError('batch size is greater than training set')
        common.shuffle_in_unison(x_set, y_set)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # for i in range(len(nabla_w)):
        #     # print(nabla_w[i].shape)
        for i in range(0, train_len, batch_size):
            x = x_set[i:i+batch_size]
            y = y_set[i:i+batch_size]
            for m in range(batch_size):
                delta_nabla_w, delta_nabla_b = self.back_prop1(x[m], y[m])
                # print('nabla_w:' + str(nabla_w[0].shape))
                # print('nabla_b:' + str(nabla_b[0].shape))
                # print([b.shape for b in delta_nabla_b])
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w-(eta/batch_size)*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/batch_size)*nb
                           for b, nb in zip(self.biases, nabla_b)]

    def train(self, x_set, y_set, batch_size=100, loop=300, eta=3):
        for i in range(loop):
            self.mini_batch(batch_size, x_set, y_set, eta)
            print('loop:%d of %d:' % (i, loop))
            self.evaluate(x_set, y_set)

    def evaluate(self, x, y):
        cost = 0
        a_list = []
        for i in range(x.shape[0]):
            a = x[i]
            # print('------------')
            for b, w in zip(self.biases, self.weights):

                # print('b:' + str(b.shape))
                a, z = self.forward(a, w, b)
                # print(a,z)
            # cost += self.cost_derivative(y[i], a)
            a_list.append(a)
        print('a.mean:%f' % a.mean())
        # print(np.array(a_list))
        corrects = np.count_nonzero(y.argmax(axis=1)==np.array(a_list).argmax(axis=1))
        print('correct:%d of %d, rate:%f'% (corrects, y.shape[0], corrects * 100 /y.shape[0]))
        # i = display_pair((x,y))
        # print(a[i].argmax())

def display_pair(mnist_set):
    x, y = mnist_set
    i = random.randint(0, x.shape[0])
    print(x[i].shape)
    print(y[i].argmax())
    imgplot = plt.imshow(x[i].reshape(28,28))
    plt.show()
    return i

n1 = Networks([784,30,10], ceptron=ceptron.Segmoid())
mn = mnist.MNIST()
x, y = mn.get_train_set(10000)
n1.train(x, y)
# display_pair((x,y))