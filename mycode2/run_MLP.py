import dataset, models, loss, optim
import numpy as np
from theano import tensor as T, function, shared

rng = np.random.RandomState(1234)


def create_mlp():
    # input layer
    x = T.matrix('x')

    # hidden layer I
    sz1 = (784, 512)
    w1 = shared(
        np.asarray(rng.uniform(low=-1, high=1, size=sz1), 'float32'),
        borrow=True)
    b1 = shared(
        np.zeros((sz1[1],), 'float32'),
        borrow=True
    )
    a1 = models.create_linear_sigmoid(x, (w1, b1))

    # hidden layer II
    sz2 = (512, 256)
    w2 = shared(
        np.asarray(rng.uniform(low=-1, high=1, size=sz2), 'float32'),
        borrow=True
    )
    b2 = shared(
        np.zeros((sz2[1],), 'float32'),
        borrow=True
    )
    a2 = models.create_linear_sigmoid(a1, (w2, b2))

    # output layer
    sz3 = (256, 10)
    tmp = rng.uniform(low=-1, high=1, size=(256, 10))
    w3 = shared(np.asarray(tmp, 'float32'), borrow=True)
    b3 = shared(np.zeros((sz3[1],), 'float32'), borrow=True)
    f = models.create_linear(a2, (w3, b3))

    return x, f, (w1, b1, w2, b2, w3, b3)


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
                 mo_create=create_mlp)
