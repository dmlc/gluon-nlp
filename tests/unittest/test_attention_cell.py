import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
from gluonnlp.model import attention_cell as ac


def check_attention_cell_basic(attention_cell, q_channel, k_channel, v_channel,
                               use_mask, multi_head=False,
                               num_heads=None):
    attention_cell.initialize()
    attention_cell.hybridize()
    for query_length, mem_length in [(10, 5), (1, 5), (5, 1)]:
        for batch_size in [1, 3]:
            if use_mask:
                mask_nd = mx.random.uniform(0, 1,
                                            shape=(batch_size, query_length, mem_length)) > 0.3
            else:
                mask_nd = None
            query_nd = mx.nd.random.normal(0, 1, (batch_size, query_length, q_channel))
            key_nd = mx.nd.random.normal(0, 1, (batch_size, mem_length, k_channel))
            value_nd = mx.nd.random.normal(0, 1, (batch_size, mem_length, v_channel))
            read_value, att_weights = attention_cell(query_nd, key_nd, value_nd, mask_nd)
            att_weights_npy = att_weights.asnumpy()
            read_value_npy = read_value.asnumpy()
            value_npy = value_nd.asnumpy()
            if not multi_head:
                if use_mask:
                    assert_allclose(att_weights_npy.sum(axis=-1),
                                    mx.nd.sum(mask_nd, axis=-1).asnumpy() > 0, 1E-5, 1E-5)
                else:
                    assert_allclose(att_weights_npy.sum(axis=-1),
                                    np.ones(att_weights.shape[:-1]), 1E-5, 1E-5)
                # Check the read value is correct
                for i in range(batch_size):
                    assert_allclose(read_value_npy[i],
                                    att_weights_npy[i].dot(value_npy[i]), 1E-5, 1E-5)
                if use_mask:
                    assert_allclose(mx.nd.norm((1 - mask_nd) * att_weights).asscalar(), 0)
            else:
                read_value_npy = read_value_npy.reshape((batch_size, query_length, num_heads,
                                                         -1))
                if use_mask:
                    mask_npy = mask_nd.asnumpy()
                for j in range(num_heads):
                    if use_mask:
                        assert_allclose(att_weights_npy[:, j, :, :].sum(axis=-1),
                                        mask_npy.sum(axis=-1) > 0, 1E-5, 1E-5)
                    else:
                        assert_allclose(att_weights_npy[:, j, :, :].sum(axis=-1),
                                        np.ones((batch_size, query_length)), 1E-5, 1E-5)
                    if use_mask:
                        assert_allclose((1 - mask_npy) * att_weights_npy[:, j, :, :], 0)


def test_mlp_attention():
    for k_channel, q_channel in [(1, 4), (4, 1), (3, 4), (4, 3), (4, 4)]:
        for use_mask in [True, False]:
            cell = ac.MLPAttentionCell(units=32)
            check_attention_cell_basic(cell, q_channel, k_channel, 5, use_mask)
            cell = ac.MLPAttentionCell(units=16, normalized=True)
            check_attention_cell_basic(cell, q_channel, k_channel, 5, use_mask)


def test_dot_product_attention():
    for k_channel, q_channel in [(1, 1), (2, 2), (4, 4)]:
        for use_mask in [True, False]:
            for scaled in [True, False]:
                for normalized in [True, False]:
                    cell = ac.DotProductAttentionCell(scaled=scaled, normalized=normalized)
                    check_attention_cell_basic(cell, q_channel, k_channel, 5, use_mask)

    for k_channel, q_channel in [(1, 2), (2, 1), (2, 4)]:
        for use_mask in [True, False]:
            for scaled in [True, False]:
                for normalized in [True, False]:
                    cell = ac.DotProductAttentionCell(units=8, scaled=scaled, normalized=normalized)
                    check_attention_cell_basic(cell, q_channel, k_channel, 5, use_mask)
                    cell = ac.DotProductAttentionCell(units=k_channel, luong_style=True,
                                                   scaled=scaled, normalized=normalized)
                    check_attention_cell_basic(cell, q_channel, k_channel, 5, use_mask)


def test_multihead_attention():
    for query_units, key_units, value_units, num_heads in [(4, 4, 8, 2), (3, 3, 9, 3),
                                                           (6, 6, 5, 1)]:
        for use_mask in [True, False]:
            for scaled in [True, False]:
                for normalized in [True, False]:
                    cell = ac.MultiHeadAttentionCell(
                        base_cell=ac.DotProductAttentionCell(scaled=scaled, normalized=normalized),
                        query_units=query_units,
                        key_units=key_units,
                        value_units=value_units,
                        num_heads=num_heads)
                    check_attention_cell_basic(cell,
                                               q_channel=query_units // num_heads,
                                               k_channel=key_units // num_heads,
                                               v_channel=value_units // num_heads,
                                               use_mask=use_mask,
                                               multi_head=True,
                                               num_heads=num_heads)
