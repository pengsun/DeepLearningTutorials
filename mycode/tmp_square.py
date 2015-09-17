from theano import tensor as T, shared, function
import numpy as np

rng = np.random.RandomState(1234)

# fprop
#p = T.matrix(name='p', dtype='float32')
p = shared(
    np.asarray(
        rng.uniform(low=-1, high=1, size=(3, 4)),
        dtype='float32'
    ),
    borrow=True
)
I = T.matrix(name='I', dtype='float32')
z = p * I
u = z**2
v = p + u
ell = T.mean(v)
# bprop
dp = T.grad(ell, wrt=p)
# the graph
fg = function(
    inputs=[I],
    outputs=[ell, v, dp],
    updates=[(p, p - np.float32(0.1)*dp)]
)

# fire
pval = np.asarray(
    rng.uniform(low=-1, high=1, size=(3, 4)),
    dtype='float32'
)
Ival = np.asarray(
    rng.uniform(low=-1, high=1, size=(3, 4)),
    dtype='float32'
)
[ellval, vval, dpval] = fg(Ival)

pass



