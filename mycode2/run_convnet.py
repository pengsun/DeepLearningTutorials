import dataset, models, loss, optim, params_init
import numpy as np
from theano import tensor as T, function, shared
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano import config

config.floatX = 'float32'
rng = np.random.RandomState(1234)
trng = RandomStreams(seed=2345)


def init_param():
    sz1_i, sz1_o = (1, 5, 5), (16,)
    sz1_pool = (2, 2)
    theta1 = [shared(tmp, borrow=True)
              for tmp in params_init.for_conv(sz1_i, sz1_o, sz1_pool, rng=rng)]

    sz2_i, sz2_o = (16, 5, 5), (16,)
    sz2_pool = (2, 2)
    theta2 = [shared(tmp, borrow=True)
              for tmp in params_init.for_conv(sz2_i, sz2_o, sz2_pool, rng=rng)]

    sz3_i, sz3_o = 256, 128
    theta3 = [shared(tmp, borrow=True)
              for tmp in params_init.for_matrix(sz3_i, sz3_o, rng=rng)]

    sz4_i, sz4_o = 128, 10
    theta4 = [shared(tmp, borrow=True)
              for tmp in params_init.for_matrix(sz4_i, sz4_o, rng=rng)]
    return theta1 + theta2 + theta3 + theta4


def create_convnet(theta, is_tr=True):
    # input
    a0 = T.matrix('x')
    a00 = a0.reshape((a0.shape[0], 1, 28, 28))

    # I: conv pool
    theta1 = [theta[0], theta[1]]
    sz1_pool = (2, 2)
    a1 = models.create_conv_pool_tanh(a00, theta1, sz1_pool)

    # II: conv pool
    theta2 = [theta[2], theta[3]]
    sz2_pool = (2, 2)
    a2 = models.create_conv_pool_tanh(a1, theta2, sz2_pool)

    # III: fc + dropout
    theta3 = [theta[4], theta[5]]
    if is_tr:
        a3 = models.create_dropout(models.create_linear_sigmoid(a2.flatten(2), theta3), trng=trng)
    else:
        a3 = models.create_dropout(models.create_linear_sigmoid(a2.flatten(2), theta3), trng=None)

    # IV: fc, output
    theta4 = [theta[6], theta[7]]
    a4 = models.create_linear(a3, theta4)

    return a0, a4


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
        pa_init=init_param,
        mo_create=create_convnet,
    )
