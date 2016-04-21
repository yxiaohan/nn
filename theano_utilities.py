import theano
import numpy as np
import theano.tensor as T


def np_array_to_shared(arr):
    return theano.shared(np.asarray(arr, dtype=theano.config.floatX))


def shared_dataset(data_set):
    x, y = data_set
    shared_x = np_array_to_shared(x)
    shared_y = np_array_to_shared(y)
    return shared_x, T.cast(shared_y, 'int32')


def shared_zeros(shape, name, borrow=True):
    return theano.shared(value=np.zeros(shape, dtype=theano.config.floatX), name=name, borrow=borrow)


def shared_randns(shape, name, borrow=True):
    return theano.shared(value=np.random.randn(*shape).astype(theano.config.floatX), name=name, borrow=borrow)


def check_gpu(theano_func):
    if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
            theano_func.maker.fgraph.toposort()]):
        print('Used the cpu')

    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
              theano_func.maker.fgraph.toposort()]):
        print('Used the gpu')

    else:
        print('ERROR, not able to tell if theano used the cpu or the gpu')
        print(theano_func.maker.fgraph.toposort())

