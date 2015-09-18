import numpy as np

dft_rng = np.random.RandomState(1234)


def for_conv(sz_i, sz_o, sz_pool=[2, 2], rng=dft_rng):
    fan_in = np.prod(sz_i)
    fan_out = np.prod(sz_o) * np.prod(sz_i[1:]) / np.prod(sz_pool)
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    w = np.asarray(
        rng.uniform(low=-bound, high=bound, size=sz_o+sz_i),
        'float32'
    )
    b = np.zeros((np.prod(sz_o),), 'float32')
    return w, b


def for_matrix(sz_i, sz_o, rng=dft_rng):
    bound = np.sqrt(6.0 / (sz_i + sz_o))
    w = np.asarray(
        rng.uniform(low=-bound, high=bound, size=[sz_i, sz_o]),
        'float32'
    )
    b = np.zeros((sz_o,), 'float32')
    return w, b
