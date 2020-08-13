import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
from mxnet import gluon
import pytest
from gluonnlp.op import *
mx.npx.set_np()


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seq_length', [16, 32])
@pytest.mark.parametrize('num_sel_positions', [1, 5])
@pytest.mark.parametrize('feature_shape', [(16,), (16, 32)])
@pytest.mark.parametrize('hybridized', [False, True])
def test_select_vectors_by_position(batch_size, seq_length, num_sel_positions,
                                    feature_shape, hybridized):
    data = mx.np.random.uniform(-1, 1, (batch_size, seq_length) + feature_shape, dtype=np.float32)
    positions = mx.np.random.randint(0, seq_length, (batch_size, num_sel_positions), dtype=np.int32)

    class Foo(gluon.HybridBlock):
        def hybrid_forward(self, F, p_data, p_positions):
            return select_vectors_by_position(F, p_data, p_positions)
    foo = Foo()
    if hybridized:
        foo.hybridize()
    out_mx = foo(data, positions)
    out_np = data.asnumpy()[np.expand_dims(np.arange(data.shape[0]).astype(np.int32),
                                           axis=1),
                            positions.asnumpy()]
    assert_allclose(out_mx.asnumpy(), out_np, 1E-4, 1E-4)


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seq_length', [16, 32])
@pytest.mark.parametrize('num_sel_positions', [1, 5])
@pytest.mark.parametrize('feature_shape,increment_shape', [((16,), (16,)),
                                                           ((16, 32), (16, 1)),
                                                           ((16, 32), (16, 32))])
@pytest.mark.parametrize('hybridized', [False, True])
def test_add_vectors_by_position(batch_size, seq_length, num_sel_positions,
                                 feature_shape, increment_shape, hybridized):
    data = mx.np.random.uniform(-1, 1, (batch_size, seq_length) + feature_shape, dtype=np.float32)
    positions = mx.np.random.randint(0, seq_length, (batch_size, num_sel_positions), dtype=np.int32)
    increment = mx.np.random.uniform(-1, 1, (batch_size, num_sel_positions) + increment_shape)

    class Foo(gluon.HybridBlock):
        def hybrid_forward(self, F, p_data, p_increment, p_positions):
            return add_vectors_by_position(F, p_data, p_increment, p_positions)

    foo = Foo()
    if hybridized:
        foo.hybridize()
    out_mx = foo(data, increment, positions).asnumpy()
    out_np = data.asnumpy().copy()
    positions = positions.asnumpy()
    increment = increment.asnumpy()
    for bidx in range(batch_size):
        for sidx in range(num_sel_positions):
            sel = positions[bidx, sidx]
            out_np[bidx, sel] += increment[bidx, sidx]
    assert_allclose(out_np, out_mx, 1E-4, 1E-4)


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seq_length', [16, 32])
@pytest.mark.parametrize('num_sel_positions', [1, 5])
@pytest.mark.parametrize('feature_shape,update_shape', [((16,), (16,)),
                                                        ((16, 32), (16, 1)),
                                                        ((16, 32), (16, 32))])
@pytest.mark.parametrize('hybridized', [False, True])
def test_update_vectors_by_position(batch_size, seq_length, num_sel_positions,
                                    feature_shape, update_shape, hybridized):
    data = mx.np.random.uniform(-1, 1, (batch_size, seq_length) + feature_shape, dtype=np.float32)
    val = mx.np.random.uniform(-1, 1, (batch_size, num_sel_positions) + update_shape)
    positions = mx.np.zeros((batch_size, num_sel_positions), dtype=np.int32)
    for i in range(batch_size):
        positions[i, :] = np.random.choice(seq_length, num_sel_positions, replace=False)

    class Foo(gluon.HybridBlock):
        def hybrid_forward(self, F, p_data, p_val, p_positions):
            return update_vectors_by_position(F, p_data, p_val, p_positions)

    foo = Foo()
    if hybridized:
        foo.hybridize()
    out_mx = foo(data, val, positions)
    out_np = data.asnumpy().copy()
    out_np[np.expand_dims(np.arange(data.shape[0]).astype(np.int32), axis=1),
           positions.asnumpy()] = val.asnumpy()
    assert_allclose(out_mx.asnumpy(), out_np, 1E-4, 1E-4)

