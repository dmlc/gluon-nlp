import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
from mxnet import gluon
from scipy.stats import ks_2samp
import pytest
from gluonnlp.op import *



@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('seq_length', [16, 32])
@pytest.mark.parametrize('num_sel_positions', [1, 5])
@pytest.mark.parametrize('feature_shape', [(16,), (16, 32)])
@pytest.mark.parametrize('hybridized', [False, True])
@pytest.mark.seed(1)
def test_select_vectors_by_position(batch_size, seq_length, num_sel_positions,
                                    feature_shape, hybridized):
    data = mx.np.random.uniform(-1, 1, (batch_size, seq_length) + feature_shape, dtype=np.float32)
    positions = mx.np.random.randint(0, seq_length, (batch_size, num_sel_positions), dtype=np.int32)

    class Foo(gluon.HybridBlock):
        def forward(self, p_data, p_positions):
            return select_vectors_by_position(p_data, p_positions)
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
@pytest.mark.seed(1)
def test_add_vectors_by_position(batch_size, seq_length, num_sel_positions,
                                 feature_shape, increment_shape, hybridized):
    data = mx.np.random.uniform(-1, 1, (batch_size, seq_length) + feature_shape, dtype=np.float32)
    positions = mx.np.random.randint(0, seq_length, (batch_size, num_sel_positions), dtype=np.int32)
    increment = mx.np.random.uniform(-1, 1, (batch_size, num_sel_positions) + increment_shape)

    class Foo(gluon.HybridBlock):
        def forward(self, p_data, p_increment, p_positions):
            return add_vectors_by_position(p_data, p_increment, p_positions)

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
@pytest.mark.seed(1)
def test_update_vectors_by_position(batch_size, seq_length, num_sel_positions,
                                    feature_shape, update_shape, hybridized):
    data = mx.np.random.uniform(-1, 1, (batch_size, seq_length) + feature_shape, dtype=np.float32)
    val = mx.np.random.uniform(-1, 1, (batch_size, num_sel_positions) + update_shape)
    positions = mx.np.zeros((batch_size, num_sel_positions), dtype=np.int32)
    for i in range(batch_size):
        positions[i, :] = np.random.choice(seq_length, num_sel_positions, replace=False)

    class Foo(gluon.HybridBlock):
        def forward(self, p_data, p_val, p_positions):
            return update_vectors_by_position(p_data, p_val, p_positions)

    foo = Foo()
    if hybridized:
        foo.hybridize()
    out_mx = foo(data, val, positions)
    out_np = data.asnumpy().copy()
    out_np[np.expand_dims(np.arange(data.shape[0]).astype(np.int32), axis=1),
           positions.asnumpy()] = val.asnumpy()
    assert_allclose(out_mx.asnumpy(), out_np, 1E-4, 1E-4)


@pytest.mark.parametrize('shape', [(10,), (5, 10)])
@pytest.mark.seed(1)
def test_gumbel_softmax(shape):
    # Here, we just verify that it will generate one-hot vectors and will have gradient
    logits = mx.np.random.uniform(-2, -1, shape)
    ret = gumbel_softmax(logits)
    assume_allones = (ret == 1).sum(axis=-1).asnumpy()
    assert_allclose(assume_allones, np.ones_like(assume_allones))

@pytest.mark.parametrize('shape', (50,))
@pytest.mark.seed(1)
def test_trunc_gumbel(shape):
    #  We first just verify that the samples are smaller than the provided threshold (i.e. they are truncated)
    #  And also attempt to remove the truncation and verify if it is sampled from a gumbel distribution
    #  using a KS-test with another sampled gumbel distribution
        
    # Verifying if the distribution is truncated
    for i in range(1000):
        samples = trunc_gumbel(mx.np.ones(shape), 1.0).asnumpy()
        assert (samples < 1.0).all()
    
    # perform ks-tests
    pvalues = []
    for i in range(1000):    
        logits = mx.np.random.uniform(-2, -1, shape)
        sampled_gumbels = mx.np.random.gumbel(mx.np.zeros_like(logits)) + logits # sample a gumbel distribution

        # sample a potential truncated gumbel distribution
        gumbels = mx.np.random.gumbel(mx.np.zeros_like(logits)) + logits
        sampled_truncated_gumbels = trunc_gumbel(logits, 0.5)
        
        # remove the truncation
        reconstructed_sample = -mx.np.log(mx.np.exp(-sampled_truncated_gumbels) - mx.np.exp(-0.5))

        pvalue = ks_2samp(reconstructed_sample.asnumpy(), sampled_gumbels.asnumpy()).pvalue
        pvalues.append(pvalue)
    
    pvalues = np.array(pvalues)
    # Statistical inference condition: if out of all the tests, 90% of the resultant p-values > 0.05, 
    # accept the null hypothesis (i.e. the reconstructed_samples indeed arrive from a gumbel distribution) 
    assert (len(pvalues[pvalues > 0.05]) > 900)
    
