import dataset, models, loss, optim
import numpy as np
from theano import tensor as T, function, shared


def init_theta():
    rng = np.random.RandomState(1234)
    szAll = [(784, 512), (512, 256), (256, 10)]
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
