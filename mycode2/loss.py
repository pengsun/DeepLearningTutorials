from theano import tensor as T


def create_logistic(F, y):
    p = T.nnet.softmax(F)
    ell = -T.mean(T.log(p)[T.arange(y.shape[0]), y])
    return ell


def create_zeroone(F, y):
    yhat = T.argmax(F, axis=1)
    return T.mean(T.neq(yhat, y))
