from theano import tensor as T, shared, function
import numpy as np


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
