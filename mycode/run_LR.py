import dataset, models, loss, optim
import numpy as np
from theano import tensor as T, function, shared


def init_theta():
    M, K = 784, 10
    theta = [None]*2
    theta[0] = shared(np.zeros((M, K), 'float32'), borrow=True)
    theta[1] = shared(np.zeros((K,), 'float32'), borrow=True)
    return theta


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
                 init_theta=init_theta,
                 mo_create=models.create_linear)