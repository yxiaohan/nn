import ceptron
import theano_base_net
import theano_cost_function
import common
import mnist


def train(net: theano_base_net.BaseNet, n_epochs, batch_size, learning_rate=0.3, auto_load_mnist=True, **kwargs):
    print('layers:{!r}'.format(net.layers))
    print('loading data sets...')
    if auto_load_mnist:
        mn = mnist.MNIST()
        train_set = mn.theano_train_set()
        valid_set = mn.theano_valid_set()
        test_set = mn.theano_test_set()
    else:
        train_set = kwargs['train_set']
        valid_set = kwargs['valid_set']
        test_set = kwargs['test_set']

    # select custom cost/loss function
    net.set_train_model(train_set, theano_cost_function.negative_log_likelihood, batch_size, learning_rate)

    # set up valid and test models, by selecting errors function
    errors = theano_cost_function.zero_one(net.p_y, net.y)
    net.set_valid_test_models(valid_set, test_set, errors)

    n_batches = net.get_batch_number(train_set, batch_size)
    print('number of batches: {0}'.format(n_batches))
    timer = common.SpeedTest()
    timer.start()
    epoch_count = 0
    last_test_score = None
    for epoch in range(n_epochs):
        print('starting epoch: %d' % epoch)
        costs = []
        for batch_index in range(n_batches):
            costs.append(net.train_model(batch_index))
        # avg_train_cost = np.mean(costs)
        # print('average training cost:%f' % avg_train_cost)

        valid_errors = net.valid_model()
        print('validation errors: %f' % valid_errors)
        epoch_count += 1
        try:
            if net.is_new_best(valid_errors):
                last_test_score = net.test_model()
                print('new best found, testing:{0}'.format(last_test_score))
                net.save_params()
        except UserWarning as e:
            print(e)
            break
    time_used = timer.stop().total_seconds()
    print('The code run for %d epochs, with %f epochs/sec' % (epoch_count, 1. * epoch_count / time_used))
    print('last test score:{0}'.format(last_test_score))
    return valid_set, test_set


def test():
    net = theano_base_net.BaseNet.net_from_layer_types((28,28), [(50, ceptron.Tanh()), (10, ceptron.TheanoSoftMax())])
    net.get_b()
    # net.save_params()
    valid_set, test_set = train(net, 1000, 600, learning_rate=0.01)
    print('after training:')
    net.get_b()
    print('test before reset:{0}'.format(net.test_model()))
    net.reset_params()

    print('test after reset:{0}'.format(net.test_model()))
    net.get_b()
    # train(net, 1000, 600, learning_rate=0.01)
    net.load_params()
    print('after loading:')
    net.get_b()
    print('load best test:{0}'.format(net.test_model()))
    print('done')


if __name__ == '__main__':
    test()
