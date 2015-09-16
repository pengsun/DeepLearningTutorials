from theano import tensor as T
import numpy as np


def update_gd(theta, dtheta, lr=np.float32(0.1)):
    return [a - lr*d for (a, d) in zip(theta, dtheta)]
