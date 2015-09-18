import dataset, models, loss, optim, params_init
import numpy as np
from theano import tensor as T, function, shared

rng = np.random.RandomState(1234)


def create_mlp():
    # input layer
    x = T.matrix('x')

    # hidden layer I
    sz1_i, sz1_o = 784, 512
    theta1 = [shared(tmp, borrow=True)
              for tmp in params_init.for_matrix(sz1_i, sz1_o, rng=rng)]
    a1 = models.create_linear_sigmoid(x, theta1)

    # hidden layer II
    sz2_i, sz2_o = 512, 256
    theta2 = [shared(tmp, borrow=True)
              for tmp in params_init.for_matrix(sz2_i, sz2_o, rng=rng)]
    a2 = models.create_linear_sigmoid(a1, theta2)

    # output layer
    sz3_i, sz3_o = 256, 10
    theta3 = [shared(tmp, borrow=True)
              for tmp in params_init.for_matrix(sz3_i, sz3_o, rng=rng)]
    a3 = models.create_linear_sigmoid(a2, theta3)

    return x, a3, theta1 + theta2 + theta3


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
