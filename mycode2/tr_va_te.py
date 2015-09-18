import dataset, models, loss, optim
import numpy as np
from theano import tensor as T, function as graph_mgr, shared


def dft_mo_create():
    n_in, n_out = 784, 10
    w = shared(np.zeros((n_in, n_out), 'float32'), borrow=True)
    b = shared(np.zeros((n_out,), 'float32'), borrow=True)

    # the dag model: simple linear model
    x = T.matrix('x', 'float32')
    f = T.dot(x, w) + b

    # outputs contract
    nodes_src = x
    nodes_sink = f
    theta = [w, b]
    return nodes_src, nodes_sink, theta


def dft_data_load():
    return dataset.load(name='mnist.pkl.gz')


def train(itMax=100, szBatch=256, lr=0.01, vaFreq=10,
          mo_create=models.create_mlp,
          data_load=dft_data_load):
    print 'loading data...'
    dataTr, dataVa, _ = data_load()

    print 'building graph...'
    # fprop: the prediction model for MLP
    x, F, theta = mo_create()
    # fprop: the loss
    y = T.ivector('y')
    ell = loss.create_logistic(F, y)
    # bprop
    dtheta = T.grad(ell, wrt=theta)
    # the graph for training
    ibat = T.lscalar('ibat')
    fg_tr = graph_mgr(
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
    fg_va = graph_mgr(
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
    plt.plot([i*vaFreq for i in range(len(er_va))], er_va, 'bx-')
    plt.show(block=True)
    # return the parameters
    return (x, F, theta)


def test(mo, data_load=dft_data_load):
    print 'loading data...'
    _, _, dataTe = data_load()

    print 'building the graph...'
    x, f, theta = mo
    # fprop zero-one loss
    y = T.ivector('y')
    ell = loss.create_zeroone(f, y)
    # all in one graph
    fg_te = graph_mgr(
        inputs=[],
        outputs=ell,
        givens={x: dataTe[0], y: dataTe[1]}
    )

    print 'fire the graph...'
    er = fg_te()
    print 'error rate = %5.4f' % (er,)


def run(itMax=195*2, szBatch=256, lr=0.1, vaFreq=20,
        mo_create=models.create_mlp,
        data_load=dft_data_load):
    print 'Training...'
    mo = train(itMax=itMax, szBatch=szBatch, lr=lr, vaFreq=vaFreq,
               mo_create=mo_create,
               data_load=data_load)

    print 'Testing...'
    test(mo)


if __name__ == '__main__':
    # config
    itMax = 195*2
    szBatch = 256
    lr = 0.1
    vaFreq = 20

    run(itMax, szBatch, lr, vaFreq)