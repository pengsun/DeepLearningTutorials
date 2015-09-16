import cPickle, gzip, os, sys
from theano import function, shared, tensor as T
import theano
import numpy as np


def load_mnist(dataset='mnist.pkl.gz'):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)


    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = shared(np.asarray(data_x, 'float32'),
                          borrow=borrow)
        shared_y = shared(np.asarray(data_y, 'float32'),
                          borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def create_model_tr(x):
    # parameter theta
    M, K = 784, 10
    w = shared(np.zeros((M, K), 'float32'),
               name='w',
               borrow=True)
    b = shared(np.zeros((K,), 'float32'),
               name='b',
               borrow=True)
    # output
    F = T.dot(x, w) + b
    #
    return F, (w, b)


def create_loss(F, y):
    p = T.nnet.softmax(F)
    ell = -T.mean(T.log(p)[T.arange(y.shape[0]), y])
    return ell


def train(itMax=100, szBatch=256):
    print 'loading data...'
    dataTr, dataVa, dataTe = load_mnist()

    print 'building the graph...'
    # fprop
    x, y = T.matrix('x', 'float32'), T.ivector('y')
    F, theta = create_model_tr(x)
    ell = create_loss(F, y)
    # bprop
    dtheta = T.grad(ell, theta)

    # SGD optimization with mini-batch
    def get_batch(iBatch, data):
        return data[iBatch*szBatch:(iBatch+1)*szBatch]

    def update_theta(theta, dtheta, lr=np.float32(0.1)):
        return [(a, a - lr*d) for (a, d) in zip(theta, dtheta)]

    # all in one graph
    iBatch = T.lscalar('iBatch')
    f_graph = function(
        [iBatch],
        outputs=ell,
        updates=update_theta(theta, dtheta),
        givens={
            x: get_batch(iBatch, dataTr[0]),
            y: get_batch(iBatch, dataTr[1])
        }
    )

    print 'begin training...'
    trLoss = []
    for i in xrange(itMax):
        tmpLoss = f_graph(i)
        print 'iteration %d, loss = %6.5f' % (i, tmpLoss)
        trLoss.append(tmpLoss)
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(trLoss)+1), trLoss, 'ro-')
    plt.show(block=True)

if __name__ == '__main__':
    train()