# make sub dir working with main dir
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from my_tensor_flow.tf_base_net import BaseNet
from my_tensor_flow.tf_layers import *
import my_tensor_flow.tf_ceptrons as tf_ceptrons

import base_layer
import mnist

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '../data/mnist', 'Directory for storing data')

# mnist1 = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


def train(net: BaseNet, n_epochs, batch_size, learning_rate=0.5, auto_load_mnist=True, **kwargs):
    print('layers:{!r}'.format(net.layers))
    print('loading data sets...')
    if auto_load_mnist:
        mn = mnist.MNIST()
        train_set = mn.tf_train_set()
        valid_set = mn.tf_valid_set()
        test_set = mn.tf_test_set()
        # train_set = (mnist1.train.images, mnist1.train.labels)
        # valid_set = (mnist1.validation.images, mnist1.validation.labels)
        # test_set = (mnist1.test.images, mnist1.test.labels)
    else:
        train_set = kwargs['train_set']
        valid_set = kwargs['valid_set']
        test_set = kwargs['test_set']

    data_sets = (train_set, valid_set, test_set)
    # select custom cost/loss function
    net.train(data_sets, batch_size, n_epochs, learning_rate)


def test():
    layers = [base_layer.DirectLayer((784,)), CeptronLayer(50, tf_ceptrons.Tanh()),
              CeptronLayer(10, tf_ceptrons.SoftMax())]
    # net = theano_base_net.BaseNet.net_from_layer_types((28,28), [(50, my_ceptron.Tanh()), (10, my_ceptron.TheanoSoftMax())])
    net = BaseNet(layers)
    # net.get_b()
    # net.save_params()
    train(net, 1000, 1000)
    # print('after training:')
    # net.get_b()
    # print('test before reset:{0}'.format(net.test_model()))
    # net.reset_params()
    #
    # print('test after reset:{0}'.format(net.test_model()))
    # net.get_b()
    # # train(net, 1000, 600, learning_rate=0.01)
    # net.load_params()
    # print('after loading:')
    # net.get_b()
    # print('load best test:{0}'.format(net.test_model()))
    print('done')


if __name__ == '__main__':
    test()
