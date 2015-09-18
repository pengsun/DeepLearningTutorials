import dataset, models, loss, optim
import numpy as np
from theano import tensor as T, function, shared
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano import config

config.floatX = 'float32'
rng = np.random.RandomState(1234)


# def data_load():
#     data = dataset.load(name='mnist.pkl.gz')
#
#     def make_featuremap(xx):
#         n = xx.shape[0]
#         return T.reshape(xx, (n, 1, 28, 28))
#
#     return [(make_featuremap(each[0]), each[1]) for each in data]


def init_param_conv(sz_i, sz_o, sz_pool=[2, 2]):
    fan_in = np.prod(sz_i)
    fan_out = np.prod(sz_o) * np.prod(sz_i[1:]) / np.prod(sz_pool)
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    w = np.asarray(
        rng.uniform(low=-bound, high=bound, size=sz_o+sz_i),
        'float32'
    )
    b = np.zeros((np.prod(sz_o),), 'float32')
    return w, b


def init_param_matrix(sz_i, sz_o):
    bound = np.sqrt(6.0 / (sz_i + sz_o))
    w = np.asarray(
        rng.uniform(low=-bound, high=bound, size=[sz_i, sz_o]),
        'float32'
    )
    b = np.zeros((sz_o,), 'float32')
    return w, b


def create_convnet():
    # input
    a0 = T.matrix('x')
    a00 = a0.reshape((a0.shape[0], 1, 28, 28))

    # I: conv pool
    sz1_i, sz1_o = (1, 5, 5), (16,)
    sz1_pool = (2, 2)
    theta1 = [shared(tmp, borrow=True) for tmp in init_param_conv(sz1_i, sz1_o, [2, 2])]
    a1 = models.create_convpool(a00, theta1, sz1_pool)

    # II: conv pool
    sz2_i, sz2_o = (16, 5, 5), (16,)
    sz2_pool = (2, 2)
    theta2 = [shared(tmp, borrow=True) for tmp in init_param_conv(sz2_i, sz2_o, [2, 2])]
    a2 = models.create_convpool(a1, theta2, sz2_pool)

    # III: fc
    sz3_i, sz3_o = 256, 128
    theta3 = [shared(tmp, borrow=True) for tmp in init_param_matrix(sz3_i, sz3_o)]
    a3 = models.create_linear_sigmoid(a2.flatten(2), theta3)

    # IV: fc, output
    sz4_i, sz4_o = 128, 10
    theta4 = [shared(tmp, borrow=True) for tmp in init_param_matrix(sz4_i, sz4_o)]
    a4 = models.create_linear(a3, theta4)

    return a0, a4, theta1 + theta2 + theta3 + theta4


if __name__ == '__main__':
    # config
    itMax = 195*2
    szBatch = 256
    lr = 0.1
    vaFreq = 20

    import tr_va_te
    tr_va_te.run(
        itMax=itMax,
        szBatch=szBatch,
        lr=lr,
        vaFreq=vaFreq,
        mo_create=create_convnet,
    )
