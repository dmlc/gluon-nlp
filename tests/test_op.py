import numpy as np
import mxnet as mx
import pytest
from gluonnlp.op import *
mx.npx.set_np()


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seq_length', [16, 32])
@pytest.mark.parametrize('num_sel_positions', [1, 5])
@pytest.mark.parametrize('feature_shape', [(16,), (16, 32)])
def test_select_vectors_by_position(batch_size, seq_length, num_sel_positions, feature_shape):
    data = mx.np.random.uniform(-1, 1, (batch_size, seq_length) + feature_shape, dtype=np.float32)
    positions = mx.np.random.randint(0, seq_length, (batch_size, num_sel_positions), dtype=np.int32)
    out = select_vectors_by_position(mx, data, positions)


