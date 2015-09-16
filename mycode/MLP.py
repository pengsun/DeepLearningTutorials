import dataset, models, loss, optim
import numpy as np
from theano import tensor as T, function, shared

def train(itMax=100, szBatch=256, lr=0.01, vaFreq=10):
    print 'loading data...'
    dataTr, dataVa, _ = dataset.load(name='mnist.pkl.gz')

    print 'building graph...'
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
    # fprop: the MLP model
    x = T.matrix('x', 'float32')
    theta = init_theta()
    F = models.create_mlp(x, theta)
    # fprop: the loss
    y = T.ivector('y')
    ell = loss.create_logistic(F, y)
    # bprop
    dtheta = T.grad(ell, wrt=theta)
    # the graph for training
    ibat = T.lscalar('ibat')
    fg_tr = function(
        inputs=[ibat],
        outputs=ell,
        updates=zip(theta, optim.update_gd(theta, dtheta)),
        givens={
            x: dataset.get_batch(ibat, dataTr[0], szBatch=szBatch),
            y: dataset.get_batch(ibat, dataTr[1], szBatch=szBatch)
        }
    )
    # the graph for validation
    ell_zo = loss.create_zeroone(F, y)
    fg_va = function(
        inputs=[],
        outputs=ell_zo,
        givens={
            x: dataVa[0],
            y: dataVa[1]
        }
    )


    print 'Fire the graph...'
    trLoss, er_va = [], []
    N = dataTr[0].get_value(borrow=True).shape[0]
    numBatch = (N + szBatch) / szBatch
    print '#batch = %d' % (numBatch,)
    for i in xrange(itMax):
        ibat = i % numBatch
        tmpLoss = fg_tr(ibat)
        print 'training: iteration %d, ibat = %d, loss = %6.5f' % (i, ibat, tmpLoss)
        trLoss.append(tmpLoss)
        if i%vaFreq == 0:
            tmp_er = fg_va()
            print 'validation: iteration %d, error rate = %6.5f' % (i, tmp_er)
            er_va.append(tmp_er)

    # plot
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(trLoss)+1), trLoss, 'ro-')
    plt.subplot(1, 2, 2)
    plt.plot([i%vaFreq for i in range(len(er_va))], er_va, 'bx-')
    plt.show(block=True)
    # return the parameters
    return theta


def test(theta):
    print 'loading data...'
    _, _, dataTe = dataset.load(name='mnist.pkl.gz')

    print 'building the graph...'
    # fprop
    x = T.matrix('x', 'float32')
    F = models.create_mlp(x, theta)
    # zero-one loss
    y = T.ivector('y')
    ell = loss.create_zeroone(F, y)
    # all in one graph
    f_graph = function(
        inputs=[],
        outputs=ell,
        givens={x: dataTe[0], y: dataTe[1]}
    )

    print 'fire the graph...'
    er = f_graph()
    print 'error rate = %5.4f' % (er,)


if __name__ == '__main__':
    # config
    itMax = 195*2
    szBatch = 256
    vaFreq = 20

    print 'Training...'
    theta = train(itMax=itMax, szBatch=szBatch, vaFreq=vaFreq)

    print 'Testing...'
    test(theta)