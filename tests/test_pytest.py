import random
import pytest
import numpy as np
import mxnet as mx


@pytest.mark.seed(1)
def test_test():
    """Test that fixing a random seed works."""
    py_rnd = random.randint(0, 100)
    np_rnd = np.random.randint(0, 100)
    mx_rnd = mx.nd.random_uniform(shape=(1, )).asscalar()

    random.seed(1)
    mx.random.seed(1)
    np.random.seed(1)

    assert py_rnd == random.randint(0, 100)
    assert np_rnd == np.random.randint(0, 100)
    assert mx_rnd == mx.nd.random_uniform(shape=(1, )).asscalar()
