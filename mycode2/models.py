from theano import tensor as T, shared, function
import numpy as np
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


def create_linear(x, theta):
    w, b = theta[0], theta[1]
    F = T.dot(x, w) + b
    return F


def create_linear_sigmoid(x, theta):
    F = create_linear(x, theta)
    a = T.nnet.sigmoid(F)
    return a


def create_mlp(x, theta):
    assert(len(theta)%2 == 0)
    numTrans = len(theta)/2
    F = [x]
    for j in xrange(numTrans):
        i = 2*j
        if j < numTrans-1:
            F.append(create_linear_sigmoid(F[j], theta[i:i+2]))
        else:
            F.append(create_linear(F[j], theta[i:i+2]))  # plain output at the last layer
    return F[-1]


def create_conv_pool_tanh(x, theta, sz_pool):
    w, b = theta

    # convolve input feature maps with filters
    a = conv.conv2d(input=x, filters=w)

    # downsample each feature map individually, using maxpooling
    aa = downsample.max_pool_2d(input=a, ds=sz_pool, ignore_border=True)

    # add the bias term. Since the bias is a vector (1D array), we first
    # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
    # thus be broadcasted across mini-batches and feature map
    # width & height
    return T.tanh(aa + b.dimshuffle('x', 0, 'x', 'x'))
