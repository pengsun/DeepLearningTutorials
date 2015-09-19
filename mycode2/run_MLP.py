import dataset, models, loss, optim, params_init
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from theano import tensor as T, function, shared

rng = np.random.RandomState(1234)
trng = RandomStreams(seed=2345)


def init_param():
    sz1_i, sz1_o = 784, 512
    theta1 = [shared(tmp, borrow=True)
              for tmp in params_init.for_matrix(sz1_i, sz1_o, rng=rng)]
    sz2_i, sz2_o = 512, 256
    theta2 = [shared(tmp, borrow=True)
              for tmp in params_init.for_matrix(sz2_i, sz2_o, rng=rng)]
    sz3_i, sz3_o = 256, 10
    theta3 = [shared(tmp, borrow=True)
              for tmp in params_init.for_matrix(sz3_i, sz3_o, rng=rng)]
    return theta1 + theta2 + theta3


def create_mlp(theta, is_tr=True):
    # input layer
    x = T.matrix('x')

    # hidden layer I
    theta1 = (theta[0], theta[1])
    if is_tr:
        a1 = models.create_dropout(models.create_linear_sigmoid(x, theta1), trng=trng)
    else:
        a1 = models.create_dropout(models.create_linear_sigmoid(x, theta1), trng=None)

    # hidden layer II
    theta2 = (theta[2], theta[3])
    if is_tr:
        a2 = models.create_dropout(models.create_linear_sigmoid(a1, theta2), trng=trng)
    else:
        a2 = models.create_dropout(models.create_linear_sigmoid(a1, theta2), trng=None)

    # output layer
    theta3 = (theta[4], theta[5])
    a3 = models.create_linear_sigmoid(a2, theta3)

    return x, a3


if __name__ == '__main__':
    # config
    itMax = 195*2
    szBatch = 256
    lr = 0.1
    vaFreq = 20

    import tr_va_te
    tr_va_te.run(itMax=itMax,
                 szBatch=szBatch,
                 lr=lr,
                 vaFreq=vaFreq,
                 pa_init=init_param,
                 mo_create=create_mlp,
    )
