import dataset, models, loss, optim
import numpy as np
from theano import tensor as T, function, shared


rng = np.random.RandomState(1234)


def init_param():
    sz3 = (784, 10)
    tmp = rng.uniform(low=-1, high=1, size=sz3)
    w3 = shared(np.zeros(sz3, 'float32'), borrow=True)
    b3 = shared(np.zeros((sz3[1],), 'float32'), borrow=True)
    return w3, b3


def create_LR(theta, is_tr=True):
    # Input
    x = T.matrix('x')

    # output layer
    w3, b3 = theta[0], theta[1]
    f = models.create_linear(x, (w3, b3))

    return x, f


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
                 mo_create=create_LR,
    )
