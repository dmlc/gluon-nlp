import numpy as np
from numpy.testing import assert_allclose
import pytest
import mxnet as mx
from mxnet.gluon import HybridBlock
from gluonnlp.attention_cell import\
    multi_head_dot_attn, gen_self_attn_mask, gen_mem_attn_mask,\
    MultiHeadAttentionCell,\
    RelAttentionScoreCell
from gluonnlp.utils.parameter import grad_global_norm
mx.npx.set_np()


@pytest.mark.parametrize('num_heads', [1, 2, 3])
@pytest.mark.parametrize('scaled', [True, False])
@pytest.mark.parametrize('normalized', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('rel_score_type', ['share_head', 'no_share_head', 'no'])
@pytest.mark.seed(123)
def test_multi_head_dot_attention_cell(num_heads, scaled, normalized, hybridize, rel_score_type, ctx):
    with ctx:
        batch_size = 5
        query_length, mem_length = 16, 32
        query_head_units = 8
        mem_head_units = 6
        query_units = query_head_units * num_heads
        mem_units = mem_head_units * num_heads
        seed = 100
        attn_cells = dict()
        for layout in ['NKT', 'NTK', 'TNK']:
            for use_einsum in [False, True]:
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
        if rel_score_type == 'share_head':
            rel_scores_np = np.random.normal(0, 1, (query_length, mem_length))
        elif rel_score_type == 'no_share_head':
            rel_scores_np = np.random.normal(0, 1, (num_heads, query_length, mem_length))
        else:
            rel_scores_np = None
        out_np = None
        score_np = None
        attn_weights_np = None
        stored_layout = None
        query_grad_np = None
        key_grad_np = None
        value_grad_np = None
        rel_scores_grad_np = None
        for (layout, use_einsum), attn_cell in attn_cells.items():
            mx.npx.random.seed(seed)
            if rel_score_type != 'no':
                rel_scores = mx.np.array(rel_scores_np, dtype=np.float32)
            else:
                rel_scores = None
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
            if rel_scores is not None:
                rel_scores.attach_grad()
            with mx.autograd.record():
                out, [score, attn_weights] = attn_cell(query, key, value, mask, rel_scores)
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
                if rel_score_type != 'no':
                    rel_scores_grad_np = rel_scores.grad.asnumpy()
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
                    if rel_score_type != 'no':
                        m_rel_scores_grad_np = rel_scores.grad.asnumpy()
                elif layout == 'NTK':
                    m_out_np = out.asnumpy()
                    m_score_np = score.asnumpy()
                    m_attn_weights_np = attn_weights.asnumpy()
                    m_query_grad_np = query.grad.asnumpy().transpose((0, 2, 1, 3))
                    m_key_grad_np = key.grad.asnumpy().transpose((0, 2, 1, 3))
                    m_value_grad_np = value.grad.asnumpy().transpose((0, 2, 1, 3))
                    if rel_score_type != 'no':
                        m_rel_scores_grad_np = rel_scores.grad.asnumpy()
                elif layout == 'TNK':
                    m_out_np = out.asnumpy().transpose((1, 0, 2))
                    m_score_np = score.asnumpy()
                    m_attn_weights_np = attn_weights.asnumpy()
                    m_query_grad_np = query.grad.asnumpy().transpose((1, 2, 0, 3))
                    m_key_grad_np = key.grad.asnumpy().transpose((1, 2, 0, 3))
                    m_value_grad_np = value.grad.asnumpy().transpose((1, 2, 0, 3))
                    if rel_score_type != 'no':
                        m_rel_scores_grad_np = rel_scores.grad.asnumpy()
                else:
                    raise NotImplementedError
                assert_allclose(m_out_np, out_np, 1E-5, 1E-5)
                assert_allclose(m_score_np, score_np, 1E-5, 1E-5)
                assert_allclose(m_attn_weights_np, attn_weights_np, 1E-5, 1E-5)
                assert_allclose(m_query_grad_np, query_grad_np, 1E-5, 1E-5)
                assert_allclose(m_key_grad_np, key_grad_np, 1E-5, 1E-5)
                assert_allclose(m_value_grad_np, value_grad_np, 1E-5, 1E-5)
                if rel_score_type != 'no':
                    assert_allclose(m_rel_scores_grad_np, rel_scores_grad_np, 1E-5, 1E-5)


@pytest.mark.parametrize('scaled', [True, False])
@pytest.mark.parametrize('normalized', [True, False])
@pytest.mark.seed(123)
def test_dot_product_attention(scaled, normalized, ctx):
    with ctx:
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
def test_gen_attn_mask(ctx):
    class GenSelfAttnMask(HybridBlock):
        def __init__(self, dtype, layout, attn_type):
            super().__init__()
            self._dtype = dtype
            self._layout = layout
            self._attn_type = attn_type

        def hybrid_forward(self, F, data, valid_length):
            return gen_self_attn_mask(F, data, valid_length,
                                      dtype=self._dtype,
                                      layout=self._layout,
                                      attn_type=self._attn_type)

    class GenMemAttnMask(HybridBlock):
        def __init__(self, dtype, layout):
            super().__init__()
            self._dtype = dtype
            self._layout = layout

        def hybrid_forward(self, F, mem, mem_valid_length, data, valid_length):
            return gen_mem_attn_mask(F, mem, mem_valid_length, data, valid_length,
                                     dtype=self._dtype, layout=self._layout)

    with ctx:
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
            mask_gen_nt = GenSelfAttnMask(dtype=np.float32, layout='NT', attn_type='full')
            mask_gen_tn = GenSelfAttnMask(dtype=np.float32, layout='TN', attn_type='full')
            if hybridize:
                mask_gen_nt.hybridize()
                mask_gen_tn.hybridize()
            mask_nt = mask_gen_nt(data, valid_length)
            mask_nt = mask_nt.asnumpy()
            mask_tn = mask_gen_tn(mx.np.swapaxes(data, 0, 1), valid_length)
            mask_tn = mask_tn.asnumpy()
            mask = mask_nt
            assert_allclose(mask_nt, mask_tn)
            for b in range(batch_size):
                v_l = valid_length.asnumpy()[b]
                for i in range(v_l):
                    assert (mask[b, i, :v_l] == 1).all()
                    assert(mask[b, i, v_l:] == 0).all()
                for i in range(v_l, query_length):
                    assert (mask[b, i, :] == 0).all()

            # Test Causal Attention Mask
            mask_gen_nt = GenSelfAttnMask(dtype=np.float32, layout='NT', attn_type='causal')
            mask_gen_tn = GenSelfAttnMask(dtype=np.float32, layout='TN', attn_type='causal')
            if hybridize:
                mask_gen_nt.hybridize()
                mask_gen_tn.hybridize()
            mask_nt = mask_gen_nt(data, valid_length)
            mask_tn = mask_gen_tn(mx.np.swapaxes(data, 0, 1), valid_length)
            assert_allclose(mask_nt.asnumpy(), mask_tn.asnumpy())
            mask = mask_nt.asnumpy()
            for b in range(batch_size):
                v_l = valid_length.asnumpy()[b]
                for i in range(v_l):
                    assert (mask[b, i, :(i + 1)] == 1).all()
                    assert (mask[b, i, (i + 1):] == 0).all()
                for i in range(v_l, query_length):
                    assert (mask[b, i, :] == 0).all()

            # Test Mem Attention Mask
            mask_gen_nt = GenMemAttnMask(dtype=np.float32, layout='NT')
            mask_gen_tn = GenMemAttnMask(dtype=np.float32, layout='TN')
            if hybridize:
                mask_gen_nt.hybridize()
                mask_gen_tn.hybridize()
            mask_nt = mask_gen_nt(mem, mem_valid_length, data, valid_length)
            mask_tn = mask_gen_tn(mx.np.swapaxes(mem, 0, 1), mem_valid_length,
                                  mx.np.swapaxes(data, 0, 1), valid_length)
            mask = mask_nt.asnumpy()
            assert_allclose(mask_nt.asnumpy(), mask_tn.asnumpy())
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
@pytest.mark.parametrize('bidirectional', [False, True])
@pytest.mark.parametrize('hybridize', [False, True])
@pytest.mark.seed(123)
def test_multi_head_rel_attn_score(num_heads, method, bidirectional, hybridize, ctx):
    batch_size = 6
    query_length = 25
    mem_length = 20
    query_head_units = 7

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
    base_score_cell = RelAttentionScoreCell(query_units=num_heads * query_head_units,
                                            num_heads=num_heads,
                                            dropout=0.0,
                                            method=method,
                                            num_buckets=num_buckets,
                                            max_distance=max_distance,
                                            layout=base_layout,
                                            use_einsum=base_use_einsum)
    base_score_cell.initialize()
    if hybridize:
        base_score_cell.hybridize()
    # Generate the data
    query = mx.np.random.normal(0, 1,
                                (batch_size, num_heads, query_length, query_head_units),
                                dtype=np.float32)
    if method != 't5':
        rel_score_grad = mx.np.random.normal(0, 1, (batch_size, num_heads, query_length, mem_length),
                                             dtype=np.float32)
    else:
        rel_score_grad = mx.np.random.normal(0, 1,
                                             (num_heads, query_length, mem_length),
                                             dtype=np.float32)
    query_positions = mx.np.arange(query_length, dtype=np.int32)
    mem_positions = mx.np.arange(mem_length, dtype=np.int32)
    rel_positions = mx.np.expand_dims(query_positions, axis=-1)\
                    - mx.np.expand_dims(mem_positions, axis=0)
    mask = mx.np.random.randint(0, 2, (batch_size, query_length, mem_length), dtype=np.int32)
    query.attach_grad()
    with mx.autograd.record():
        rel_score = base_score_cell(rel_positions, query)
        rel_score.backward(rel_score_grad)
    original_rel_score = rel_score.asnumpy()
    original_grad_norm = grad_global_norm(base_score_cell.collect_params().values())
    original_query_grad_norm = np.linalg.norm(query.grad.asnumpy())
    assert original_grad_norm > 0
    # 1. Test for permutation equivariant
    # We can permutate the query, rel_positions and the rel_score_grad and the result should
    # always be the same.
    query_perm = mx.np.array(np.random.permutation(query_length), dtype=np.int32)
    mem_perm = mx.np.array(np.random.permutation(mem_length, ), dtype=np.int32)

    query.grad[:] = 0
    with mx.autograd.record():
        rel_score = base_score_cell(rel_positions[query_perm, :][:, mem_perm],
                                    query[:, :, query_perm, :])
        if method != 't5':
            rel_score.backward(rel_score_grad[:, :, query_perm, :][:, :, :, mem_perm])
        else:
            rel_score.backward(rel_score_grad[:, query_perm, :][:, :, mem_perm])
    permutated_out = rel_score.asnumpy()
    permutated_grad_norm = grad_global_norm(base_score_cell.collect_params().values())
    permutated_query_grad_norm = np.linalg.norm(query.grad.asnumpy())
    if method != 't5':
        assert_allclose(
            original_rel_score[:, :, query_perm.asnumpy(), :][:, :, :, mem_perm.asnumpy()],
            permutated_out, 1E-4, 1E-4)
    else:
        assert_allclose(original_rel_score[:, query_perm.asnumpy(), :][:, :, mem_perm.asnumpy()],
                        permutated_out, 1E-4, 1E-4)
    assert_allclose(permutated_grad_norm, original_grad_norm, 1E-4, 1E-4)
    assert_allclose(permutated_query_grad_norm, original_query_grad_norm, 1E-4, 1E-4)
    # 2. Test for different layout + use/not use einsum
    for layout in ['NKT', 'NTK', 'TNK']:
        for use_einsum in [False, True]:
            if layout == base_layout and use_einsum == base_use_einsum:
                continue
            score_cell = RelAttentionScoreCell(query_units=num_heads * query_head_units,
                                               num_heads=num_heads,
                                               dropout=0.0,
                                               method=method,
                                               num_buckets=num_buckets,
                                               max_distance=max_distance,
                                               layout=layout,
                                               use_einsum=use_einsum)
            score_cell.initialize()
            if hybridize:
                score_cell.hybridize()
            score_cell.load_dict({name: param.data() for name, param in base_score_cell.collect_params().items()})
            query.attach_grad()
            query.grad[:] = 0
            with mx.autograd.record():
                if layout == 'NKT':
                    rel_score = score_cell(rel_positions, query)
                    rel_score.backward(rel_score_grad)
                elif layout == 'NTK':
                    rel_score = score_cell(rel_positions, query.transpose((0, 2, 1, 3)))
                    rel_score.backward(rel_score_grad)
                elif layout == 'TNK':
                    rel_score = score_cell(rel_positions, query.transpose((2, 0, 1, 3)))
                    rel_score.backward(rel_score_grad)
                else:
                    raise NotImplementedError
            assert_allclose(rel_score.asnumpy(), original_rel_score, 1E-5, 1E-5)
            layout_query_grad_norm = np.linalg.norm(query.grad.asnumpy())
            assert_allclose(layout_query_grad_norm, original_query_grad_norm, 1E-5, 1E-5)
