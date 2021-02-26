import numpy as np
from numpy.testing import assert_allclose
import torch as th
from gluonnlp.torch import attention_cell as ac


def check_attention_cell_basic(attention_cell, q_channel, k_channel, v_channel, use_mask,
                               multi_head=False, num_heads=None, layout='NTK'):
    for query_length, mem_length in [(10, 5), (1, 5), (5, 1)]:
        for batch_size in [1, 3]:
            if use_mask:
                mask_nd = th.rand(batch_size, query_length, mem_length) > 0.3
            else:
                mask_nd = None
            if layout == 'NTK':
                query_nd = th.randn(batch_size, query_length, num_heads, q_channel)
                key_nd = th.randn(batch_size, mem_length, num_heads, k_channel)
                value_nd = th.randn(batch_size, mem_length, num_heads, v_channel)
            elif layout == 'NKT':
                query_nd = th.randn(batch_size, num_heads, query_length, q_channel)
                key_nd = th.randn(batch_size, num_heads, mem_length, k_channel)
                value_nd = th.randn(batch_size, num_heads, mem_length, v_channel)
            elif layout == 'TNK':
                query_nd = th.randn(query_length, batch_size, num_heads, q_channel)
                key_nd = th.randn(mem_length, batch_size, num_heads, k_channel)
                value_nd = th.randn(mem_length, batch_size, num_heads, v_channel)
            read_value, (scores, att_weights) = attention_cell(query_nd, key_nd, value_nd, mask_nd)
            att_weights_npy = att_weights.numpy()
            read_value_npy = read_value.numpy()
            value_npy = value_nd.numpy()
            if not multi_head:
                if use_mask:
                    assert_allclose(att_weights_npy.sum(axis=-1),
                                    th.sum(mask_nd, dim=-1).numpy() > 0, 1E-5, 1E-5)
                else:
                    assert_allclose(att_weights_npy.sum(axis=-1), np.ones(att_weights.shape[:-1]),
                                    1E-5, 1E-5)
                # Check the read value is correct
                for i in range(batch_size):
                    assert_allclose(read_value_npy[i], att_weights_npy[i].dot(value_npy[i]), 1E-5,
                                    1E-5)
                if use_mask:
                    assert_allclose(th.norm((1 - mask_nd) * att_weights).asscalar(), 0)
            else:
                read_value_npy = read_value_npy.reshape((batch_size, query_length, num_heads, -1))
                if use_mask:
                    mask_npy = mask_nd.numpy()
                for j in range(num_heads):
                    if use_mask:
                        assert_allclose(att_weights_npy[:, j, :, :].sum(axis=-1),
                                        mask_npy.sum(axis=-1) > 0, 1E-5, 1E-5)
                    else:
                        assert_allclose(att_weights_npy[:, j, :, :].sum(axis=-1),
                                        np.ones((batch_size, query_length)), 1E-5, 1E-5)
                    if use_mask:
                        assert_allclose((1 - mask_npy) * att_weights_npy[:, j, :, :], 0)


def test_multihead_attention():
    for query_units, key_units, value_units, num_heads in [(4, 4, 8, 2), (3, 3, 9, 3),
                                                           (6, 6, 5, 1)]:
        for use_mask in [True, False]:
            for scaled in [True, False]:
                for normalized in [True, False]:
                    for layout in ['NKT', 'NTK', 'TNK']:
                        cell = ac.MultiHeadAttentionCell(query_units=query_units,
                                                         num_heads=num_heads, scaled=scaled,
                                                         normalized=normalized, layout=layout)
                        check_attention_cell_basic(cell, q_channel=query_units // num_heads,
                                                   k_channel=key_units // num_heads,
                                                   v_channel=value_units // num_heads,
                                                   use_mask=use_mask, multi_head=True,
                                                   num_heads=num_heads, layout=layout)
