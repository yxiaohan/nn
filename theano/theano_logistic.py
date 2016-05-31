import numpy as np
import theano_base_net

import common
import mnist
from theano import theano_cost_function, my_ceptron


class Logistic(theano_base_net.BaseNet):
    def __init__(self, layers):
        super().__init__(layers)

    def set_train_model(self, train_set, batch_size, learning_rate):
        # select custom cost/loss function
        cost = theano_cost_function.negative_log_likelihood(self.p_y, self.y) \
               + self.l1 * 0.0 + self.l2 * 0.0001
        super()._set_train_model(train_set, cost, batch_size, learning_rate)

    def train(self, n_epochs, batch_size, learning_rate=0.3, auto_load_mnist=True, **kwargs):
        print('layers:{!r}'.format(self.layers))
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

        self.set_train_model(train_set, batch_size, learning_rate)
        # set up valid and test models, by selecting errors function
        errors = theano_cost_function.zero_one(self.p_y, self.y)
        self.valid_model = self._set_model(valid_set, errors)
        self.test_model = self._set_model(test_set, errors)

        n_batches = self.get_batch_number(train_set, batch_size)
        print('number of batches: {0}'.format(n_batches))
        timer = common.SpeedTest()
        timer.start()
        epoch_count = 0
        last_test_score = None
        for epoch in range(n_epochs):
            print('starting epoch: %d' % epoch)
            costs = []
            for batch_index in range(n_batches):
                costs.append(self.train_model(batch_index))
            avg_train_cost = np.mean(costs)
            # print('average training cost:%f' % avg_train_cost)

            valid_errors = self.valid_model()
            print('validation errors: %f' % valid_errors)
            epoch_count += 1
            try:
                if self.is_new_best(valid_errors):
                    last_test_score = self.test_model()
                    print('new best found, testing:{0}'.format(last_test_score))
                    self.save_params()
            except UserWarning as e:
                print(e)
                break
        time_used = timer.stop().total_seconds()
        print('The code run for %d epochs, with %f epochs/sec' % (epoch_count, 1. * epoch_count / time_used))
        print('last test score:{0}'.format(last_test_score))


lo = Logistic([(28 * 28, None), (50, my_ceptron.Tanh()), (10, my_ceptron.TheanoSoftMax())])
# lo.get_b()
# lo.save_params()
lo.train(1000, 600, learning_rate=0.01)
print('after training:')
# lo.get_b()
lo.reset_params()
print('test after reset:{0}'.format(lo.test_model()))
# lo.get_b()
lo.load_params()
print('after loading:')
# lo.get_b()
print('load best test:{0}'.format(lo.test_model()))
print('done')
