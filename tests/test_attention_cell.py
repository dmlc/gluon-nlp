import numpy as np
from numpy.testing import assert_allclose
import pytest
import mxnet as mx
from mxnet.gluon import HybridBlock
from gluonnlp.attention_cell import\
    multi_head_dot_attn, gen_self_attn_mask, gen_mem_attn_mask,\
    MultiHeadAttentionCell,\
    MultiHeadRelAttentionCell
mx.npx.set_np()


@pytest.mark.parametrize('num_heads', [1, 2, 3])
@pytest.mark.parametrize('scaled', [True, False])
@pytest.mark.parametrize('normalized', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.seed(123)
def test_multi_head_dot_attention_cell(num_heads, scaled, normalized, hybridize):
    batch_size = 5
    query_length, mem_length = 16, 32
    query_head_units = 8
    mem_head_units = 6
    query_units = query_head_units * num_heads
    mem_units = mem_head_units * num_heads
    seed = 100
    attn_cells = dict()
    for layout in ['NKT', 'NTK', 'TNK']:
        # TODO(sxjscience) Currently, the einsum test is disabled due to the wrong gradient issue
        #  See https://github.com/apache/incubator-mxnet/issues/18102
        for use_einsum in [False]:
            attn_cells[(layout, use_einsum)] = MultiHeadAttentionCell(
                query_units=query_units,
                num_heads=num_heads,
                attention_dropout=0.0,
                scaled=scaled,
                normalized=normalized,
                layout=layout,
                use_einsum=use_einsum)
            if hybridize:
                attn_cells[(layout, use_einsum)].hybridize()
    # Generate the data
    query_np = np.random.normal(0, 1, (batch_size, num_heads, query_length, query_head_units))
    key_np = np.random.normal(0, 1, (batch_size, num_heads, mem_length, query_head_units))
    value_np = np.random.normal(0, 1, (batch_size, num_heads, mem_length, mem_head_units))
    mask_np = np.random.randint(0, 2, (batch_size, query_length, mem_length))
    out_np = None
    score_np = None
    attn_weights_np = None
    stored_layout = None
    query_grad_np = None
    key_grad_np = None
    value_grad_np = None
    for (layout, use_einsum), attn_cell in attn_cells.items():
        mx.npx.random.seed(seed)
        if layout == 'NKT':
            query = mx.np.array(query_np, dtype=np.float32)
            key = mx.np.array(key_np, dtype=np.float32)
            value = mx.np.array(value_np, dtype=np.float32)
        elif layout == 'NTK':
            query = mx.np.array(query_np.transpose((0, 2, 1, 3)), dtype=np.float32)
            key = mx.np.array(key_np.transpose((0, 2, 1, 3)), dtype=np.float32)
            value = mx.np.array(value_np.transpose((0, 2, 1, 3)), dtype=np.float32)
        elif layout == 'TNK':
            query = mx.np.array(query_np.transpose((2, 0, 1, 3)), dtype=np.float32)
            key = mx.np.array(key_np.transpose((2, 0, 1, 3)), dtype=np.float32)
            value = mx.np.array(value_np.transpose((2, 0, 1, 3)), dtype=np.float32)
        else:
            raise NotImplementedError
        mask = mx.np.array(mask_np, dtype=np.int32)
        query.attach_grad()
        key.attach_grad()
        value.attach_grad()
        with mx.autograd.record():
            out, [score, attn_weights] = attn_cell(query, key, value, mask)
            out.backward()
        if layout == 'NKT':
            assert out.shape == (batch_size, query_length, num_heads * mem_head_units)
        elif layout == 'NTK':
            assert out.shape == (batch_size, query_length, num_heads * mem_head_units)
        elif layout == 'TNK':
            assert out.shape == (query_length, batch_size, num_heads * mem_head_units)
        else:
            raise NotImplementedError
        for i in range(num_heads):
            assert_allclose(attn_weights[:, i, :, :][mask == 0].asnumpy(),
                            mask[mask == 0].astype(np.float32).asnumpy(), 1E-5, 1E-5)

        if stored_layout is None:
            out_np = out.asnumpy()
            score_np = score.asnumpy()
            attn_weights_np = attn_weights.asnumpy()
            stored_layout = layout
            query_grad_np = query.grad.asnumpy()
            key_grad_np = key.grad.asnumpy()
            value_grad_np = value.grad.asnumpy()
        else:
            assert stored_layout == 'NKT'
            # Begin to match the output
            if layout == 'NKT':
                m_out_np = out.asnumpy()
                m_score_np = score.asnumpy()
                m_attn_weights_np = attn_weights.asnumpy()
                m_query_grad_np = query.grad.asnumpy()
                m_key_grad_np = key.grad.asnumpy()
                m_value_grad_np = value.grad.asnumpy()
            elif layout == 'NTK':
                m_out_np = out.asnumpy()
                m_score_np = score.asnumpy()
                m_attn_weights_np = attn_weights.asnumpy()
                m_query_grad_np = query.grad.asnumpy().transpose((0, 2, 1, 3))
                m_key_grad_np = key.grad.asnumpy().transpose((0, 2, 1, 3))
                m_value_grad_np = value.grad.asnumpy().transpose((0, 2, 1, 3))
            elif layout == 'TNK':
                m_out_np = out.asnumpy().transpose((1, 0, 2))
                m_score_np = score.asnumpy()
                m_attn_weights_np = attn_weights.asnumpy()
                m_query_grad_np = query.grad.asnumpy().transpose((1, 2, 0, 3))
                m_key_grad_np = key.grad.asnumpy().transpose((1, 2, 0, 3))
                m_value_grad_np = value.grad.asnumpy().transpose((1, 2, 0, 3))
            else:
                raise NotImplementedError
            assert_allclose(m_out_np, out_np, 1E-5, 1E-5)
            assert_allclose(m_score_np, score_np, 1E-5, 1E-5)
            assert_allclose(m_attn_weights_np, attn_weights_np, 1E-5, 1E-5)
            assert_allclose(m_query_grad_np, query_grad_np, 1E-5, 1E-5)
            assert_allclose(m_key_grad_np, key_grad_np, 1E-5, 1E-5)
            assert_allclose(m_value_grad_np, value_grad_np, 1E-5, 1E-5)


@pytest.mark.parametrize('scaled', [True, False])
@pytest.mark.parametrize('normalized', [True, False])
@pytest.mark.seed(123)
def test_dot_product_attention(scaled, normalized):
    num_heads = 4
    batch_size = 32
    query_length, mem_length = 16, 32
    num_channel = 8
    query = mx.np.random.normal(0, 1, (batch_size, num_heads, query_length, num_channel))
    key = mx.np.random.normal(0, 1, (batch_size, num_heads, mem_length, num_channel))
    value = mx.np.random.normal(0, 1, (batch_size, num_heads, mem_length, num_channel))
    mask = mx.np.random.randint(0, 2, (batch_size, query_length, mem_length))
    out, [score, attn_weights] = multi_head_dot_attn(mx.nd, query, key, value, mask,
                                                     scaled=scaled, normalized=normalized)
    assert out.shape == (batch_size, query_length, num_heads * num_channel)
    for i in range(num_heads):
        assert_allclose(attn_weights[:, i, :, :][mask == 0].asnumpy(),
                        mask[mask == 0].astype(np.float32).asnumpy(), 1E-5, 1E-5)


@pytest.mark.seed(123)
def test_gen_attn_mask():
    class GenSelfAttnMask(HybridBlock):
        def __init__(self, dtype, attn_type, prefix=None, params=None):
            super(GenSelfAttnMask, self).__init__(prefix=prefix, params=params)
            self._dtype = dtype
            self._attn_type = attn_type

        def hybrid_forward(self, F, data, valid_length):
            return gen_self_attn_mask(F, data, valid_length,
                                      dtype=self._dtype, attn_type=self._attn_type)

    class GenMemAttnMask(HybridBlock):
        def __init__(self, dtype, prefix=None, params=None):
            super(GenMemAttnMask, self).__init__(prefix=prefix, params=params)
            self._dtype = dtype

        def hybrid_forward(self, F, mem, mem_valid_length, data, valid_length):
            return gen_mem_attn_mask(F, mem, mem_valid_length, data, valid_length,
                                     dtype=self._dtype)

    batch_size = 4
    query_length = 8
    mem_length = 6
    nchannel = 5
    data = mx.np.random.normal(0, 1, (batch_size, query_length, nchannel), dtype=np.float32)
    valid_length = mx.np.random.randint(1, query_length + 1, (batch_size,))

    mem = mx.np.random.normal(0, 1, (batch_size, mem_length, nchannel), dtype=np.float32)
    mem_valid_length = mx.np.random.randint(1, mem_length + 1, (batch_size,))

    for hybridize in [False, True]:
        # Test Full Attention Mask
        mask_gen = GenSelfAttnMask(dtype=np.float32, attn_type='full')
        if hybridize:
            mask_gen.hybridize()
        mask = mask_gen(data, valid_length)
        mask = mask.asnumpy()
        for b in range(batch_size):
            v_l = valid_length.asnumpy()[b]
            for i in range(v_l):
                assert (mask[b, i, :v_l] == 1).all()
                assert(mask[b, i, v_l:] == 0).all()
            for i in range(v_l, query_length):
                assert (mask[b, i, :] == 0).all()

        # Test Causal Attention Mask
        mask_gen = GenSelfAttnMask(dtype=np.float32, attn_type='causal')
        if hybridize:
            mask_gen.hybridize()
        mask = mask_gen(data, valid_length)
        mask = mask.asnumpy()
        for b in range(batch_size):
            v_l = valid_length.asnumpy()[b]
            for i in range(v_l):
                assert (mask[b, i, :(i + 1)] == 1).all()
                assert (mask[b, i, (i + 1):] == 0).all()
            for i in range(v_l, query_length):
                assert (mask[b, i, :] == 0).all()

        # Test Mem Attention Mask
        mask_gen = GenMemAttnMask(dtype=np.float32)
        if hybridize:
            mask_gen.hybridize()
        mask = mask_gen(mem, mem_valid_length, data, valid_length)
        mask = mask.asnumpy()
        for b in range(batch_size):
            data_v_l = valid_length.asnumpy()[b]
            mem_v_l = mem_valid_length.asnumpy()[b]
            for i in range(data_v_l):
                assert (mask[b, i, :mem_v_l] == 1).all()
                assert (mask[b, i, mem_v_l:] == 0).all()
            for i in range(data_v_l, query_length):
                assert (mask[b, i, :] == 0).all()


@pytest.mark.parametrize('num_heads', [1, 2, 3])
@pytest.mark.parametrize('method', ['transformer_xl', 'shaw', 't5'])
@pytest.mark.parametrize('query_add_bias', [False, True, None])
@pytest.mark.parametrize('bidirectional', [False, True])
@pytest.mark.parametrize('hybridize', [False, True])
@pytest.mark.seed(123)
def test_multi_head_rel_dot_attn(num_heads, method, query_add_bias, bidirectional, hybridize):
    batch_size = 6
    query_length = 25
    mem_length = 20
    query_head_units = 7
    mem_head_units = 5

    # Initialize the attention cell with relative positional embedding
    base_layout = 'NKT'
    base_use_einsum = False
    if method == 'shaw':
        num_buckets = None
        max_distance = 20
    elif method == 't5':
        num_buckets = 10
        max_distance = 20
    elif method == 'transformer_xl':
        num_buckets = None
        max_distance = None
    else:
        raise NotImplementedError
    base_attn_cell = MultiHeadRelAttentionCell(query_units=num_heads * query_head_units,
                                               num_heads=num_heads,
                                               dropout=0.0,
                                               attention_dropout=0.0,
                                               query_add_bias=query_add_bias,
                                               method=method,
                                               num_buckets=num_buckets,
                                               max_distance=max_distance,
                                               layout=base_layout,
                                               use_einsum=base_use_einsum)
    base_attn_cell.initialize()
    if hybridize:
        base_attn_cell.hybridize()
    # Generate the data
    query = mx.np.random.normal(0, 1,
                                (batch_size, num_heads, query_length, query_head_units),
                                dtype=np.float32)
    value = mx.np.random.normal(0, 1,
                                (batch_size, num_heads, mem_length, mem_head_units),
                                dtype=np.float32)
    key = mx.np.random.normal(0, 1,
                              (batch_size, num_heads, mem_length, query_head_units),
                              dtype=np.float32)
    out_grad = mx.np.random.normal(0, 1, (batch_size, query_length, num_heads * mem_head_units),
                                   dtype=np.float32)
    query_positions = mx.np.arange(query_length, dtype=np.int32)
    mem_positions = mx.np.arange(mem_length, dtype=np.int32)
    rel_positions = mx.np.expand_dims(query_positions, axis=-1) - mx.np.expand_dims(mem_positions,
                                                                                    axis=0)
    mask = mx.np.random.randint(0, 2, (batch_size, query_length, mem_length), dtype=np.int32)
    query.attach_grad()
    key.attach_grad()
    value.attach_grad()
    with mx.autograd.record():
        out = base_attn_cell(query, key, value, rel_positions, mask)
        out[0].backward(out_grad)
    original_out = out[0].asnumpy()
    original_query_grad = query.grad.asnumpy()
    original_key_grad = key.grad.asnumpy()
    original_value_grad = value.grad.asnumpy()
    assert np.linalg.norm(original_query_grad) > 0
    assert np.linalg.norm(original_key_grad) > 0
    assert np.linalg.norm(original_value_grad) > 0
    # 1. Test for permutation equivariant
    # We can permutate the query, key, value, rel_positions and the result should
    # always be the same.
    query_perm = mx.np.array(np.random.permutation(query_length), dtype=np.int32)
    mem_perm = mx.np.array(np.random.permutation(mem_length, ), dtype=np.int32)

    query.grad[:] = 0
    key.grad[:] = 0
    value.grad[:] = 0
    with mx.autograd.record():
        out = base_attn_cell(query[:, :, query_perm, :],
                             key[:, :, mem_perm, :],
                             value[:, :, mem_perm, :],
                             rel_positions[query_perm, :][:, mem_perm],
                             mask[:, query_perm, :][:, :, mem_perm])
        out[0].backward(out_grad[:, query_perm, :])
    permutated_out = out[0].asnumpy()
    query_grad_after_perm = query.grad.asnumpy()
    key_grad_after_perm = key.grad.asnumpy()
    value_grad_after_perm = value.grad.asnumpy()
    assert_allclose(permutated_out, original_out[:, query_perm.asnumpy(), :], 1E-4, 1E-4)
    assert_allclose(query_grad_after_perm, original_query_grad, 1E-4, 1E-4)
    assert_allclose(key_grad_after_perm, original_key_grad, 1E-4, 1E-4)
    assert_allclose(value_grad_after_perm, original_value_grad, 1E-4, 1E-4)

    # 2. Test for different layout + use/not use einsum
    for layout in ['NKT', 'NTK', 'TNK']:
        # TODO(sxjscience) Currently, the einsum test is disabled due to the wrong gradient issue
        #  See https://github.com/apache/incubator-mxnet/issues/18102
        for use_einsum in [False]:
            if layout == base_layout and use_einsum == base_use_einsum:
                continue
            attn_cell = MultiHeadRelAttentionCell(query_units=num_heads * query_head_units,
                                                  num_heads=num_heads,
                                                  dropout=0.0,
                                                  attention_dropout=0.0,
                                                  query_add_bias=query_add_bias,
                                                  method=method,
                                                  num_buckets=num_buckets,
                                                  max_distance=max_distance,
                                                  layout=layout,
                                                  use_einsum=use_einsum,
                                                  params=base_attn_cell.collect_params())

            if hybridize:
                attn_cell.hybridize()
            query.attach_grad()
            key.attach_grad()
            value.attach_grad()
            query.grad[:] = 0
            key.grad[:] = 0
            value.grad[:] = 0
            with mx.autograd.record():
                if layout == 'NKT':
                    out = attn_cell(query, key, value, rel_positions, mask)
                    out[0].backward(out_grad)
                    test_out_np = out[0].asnumpy()
                elif layout == 'NTK':
                    out = attn_cell(query.transpose((0, 2, 1, 3)),
                                    key.transpose((0, 2, 1, 3)),
                                    value.transpose((0, 2, 1, 3)), rel_positions, mask)
                    out[0].backward(out_grad)
                    test_out_np = out[0].asnumpy()
                elif layout == 'TNK':
                    out = attn_cell(query.transpose((2, 0, 1, 3)),
                                    key.transpose((2, 0, 1, 3)),
                                    value.transpose((2, 0, 1, 3)), rel_positions, mask)
                    out[0].backward(out_grad.transpose((1, 0, 2)))
                    test_out_np = out[0].asnumpy().transpose((1, 0, 2))
                else:
                    raise NotImplementedError
            test_query_grad_np = query.grad.asnumpy()
            test_key_grad_np = key.grad.asnumpy()
            test_value_grad_np = value.grad.asnumpy()
            assert_allclose(test_out_np, original_out, 1E-5, 1E-5)
            assert_allclose(test_query_grad_np, original_query_grad, 1E-5, 1E-5)
            assert_allclose(test_key_grad_np, original_key_grad, 1E-5, 1E-5)
            assert_allclose(test_value_grad_np, original_value_grad, 1E-5, 1E-5)
