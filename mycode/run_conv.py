import dataset, models, loss, optim
import numpy as np
from theano import tensor as T, function, shared

rng = np.random.RandomState(1234)


def init_theta_fc():
    szAll = [(256, 128), (128, 10)]
    tmp = []
    for sz in szAll:
        tmp.append(
            np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (sz[0] + sz[1])),
                    high=np.sqrt(6. / (sz[0] + sz[1])),
                    size=sz
                ),
                dtype='float32'
            )
        )
        tmp.append(np.zeros((sz[1],), 'float32'))
    sh_theta = []
    for each in tmp:
        sh_theta.append(shared(each, borrow=True))
    return sh_theta


def init_theta():
    sz_flt = [(16, 1, 5, 5), (16, 16, 5, 5) ]
    tmp = []
    for sz in szAll:
        tmp.append(
            np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (sz[0] + sz[1])),
                    high=np.sqrt(6. / (sz[0] + sz[1])),
                    size=sz
                ),
                dtype='float32'
            )
        )
        tmp.append(np.zeros((sz[1],), 'float32'))
    sh_theta = []
    for each in tmp:
        sh_theta.append(shared(each, borrow=True))
    return sh_theta


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
                 mo_create=models.create_mlp)
