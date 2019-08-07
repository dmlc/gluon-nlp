# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test Transformer-XL."""
import functools
import re

import mxnet as mx
import pytest

import gluonnlp as nlp
from gluonnlp.model.transformer import _position_encoding_init

from ..language_model.transformer import (AdaptiveEmbedding, AdaptiveLogSoftmaxWithLoss,
                                          PositionalEmbeddingMultiHeadAttentionCell, TransformerXL)


@pytest.mark.parametrize('d_head', [5])
@pytest.mark.parametrize('num_heads', [1, 3])
@pytest.mark.parametrize('query_length', [10, 5, 1])
@pytest.mark.parametrize('memory_length', [5, 1])
@pytest.mark.parametrize('batch_size', [3, 1])
@pytest.mark.parametrize('use_mask', [True, False])
def test_positional_embedding_multihead_attention_cell(d_head, num_heads, query_length,
                                                       memory_length, batch_size, use_mask):
    attention_cell = PositionalEmbeddingMultiHeadAttentionCell(d_head=d_head, num_heads=num_heads,
                                                               scaled=False, dropout=0)
    attention_cell.initialize()

    if use_mask:
        mask_nd = mx.random.uniform(0, 1, shape=(batch_size, query_length, memory_length)) > 0.3
    else:
        mask_nd = None
    query_nd = mx.nd.random.normal(0, 1, (batch_size, query_length, d_head))
    key_nd = mx.nd.random.normal(0, 1, (batch_size, memory_length, d_head))
    value_nd = mx.nd.random.normal(0, 1, (batch_size, memory_length, d_head))
    emb_nd = mx.nd.array(_position_encoding_init(memory_length, d_head))

    read_value, att_weights = attention_cell(query_nd, key_nd, value_nd, emb_nd, mask_nd)


@pytest.mark.parametrize('embed_size', [64, 32])
@pytest.mark.parametrize('units', [64, 32])
@pytest.mark.parametrize('cutoffs', [[10], [10, 30]])
@pytest.mark.parametrize('div_val', [1, 2, 4])
@pytest.mark.parametrize('hybridize', [True, False])
def test_adaptive_embedding(embed_size, units, cutoffs, div_val, hybridize):
    vocab_size = 100
    emb = AdaptiveEmbedding(vocab_size=vocab_size, embed_size=embed_size, units=units,
                            cutoffs=cutoffs, div_val=div_val)
    emb.initialize()
    if hybridize:
        emb.hybridize()

    x = mx.nd.arange(vocab_size)
    _ = emb(x)
    mx.nd.waitall()


@pytest.mark.parametrize('embed_size', [64, 32])
@pytest.mark.parametrize('units', [64, 32])
@pytest.mark.parametrize('cutoffs', [[10], [10, 30]])
@pytest.mark.parametrize('div_val', [2, 4])
@pytest.mark.parametrize('tie_with_adaptive_embedding', [False, True])
@pytest.mark.parametrize('hybridize', [True, False])
def test_adaptive_softmax(embed_size, units, cutoffs, div_val, tie_with_adaptive_embedding,
                          hybridize):
    vocab_size = 100

    Net = functools.partial(AdaptiveLogSoftmaxWithLoss, vocab_size=vocab_size,
                            embed_size=embed_size, units=units, cutoffs=cutoffs, div_val=div_val)

    if tie_with_adaptive_embedding:
        # Share all parameters
        emb = AdaptiveEmbedding(vocab_size=vocab_size, embed_size=embed_size, units=units,
                                cutoffs=cutoffs, div_val=div_val)
        emb_params = emb.collect_params()
        net = Net(tie_embeddings=True, tie_projections=[True] * (len(cutoffs) + 1),
                  params=emb_params)
        for param_name, param in net.collect_params().items():
            if re.search(r'(?:(?:embedding)|(?:projection))\d+_weight', param_name):
                assert param in emb_params.values()
            elif re.search(r'(?:(?:embedding)|(?:projection))\d+_bias', param_name):
                assert param not in emb_params.values()

        # Share only embedding parameters
        net = Net(tie_embeddings=True, params=emb_params)
        for param_name, param in net.collect_params().items():
            if re.search(r'(?:embedding)\d+_weight', param_name):
                assert param in emb_params.values()
            elif re.search(r'(?:projection)|(?:bias)', param_name):
                assert param not in emb_params.values()
    else:
        net = Net()

    net.initialize()
    if hybridize:
        net.hybridize()

    x = mx.nd.random.normal(shape=(8, 16, units))
    y = mx.nd.arange(8 * 16).clip(0, vocab_size - 1).reshape((8, 16))
    _ = net(x, y)
    mx.nd.waitall()


@pytest.mark.parametrize('embed_size', [64, 32])
@pytest.mark.parametrize('units', [64, 32])
@pytest.mark.parametrize('cutoffs', [[10], [10, 30]])
@pytest.mark.parametrize('div_val', [1, 2, 4])
@pytest.mark.parametrize('mem_len', [8, 16])
@pytest.mark.parametrize('hybridize', [True, False])
def test_transformer_xl_model(embed_size, units, cutoffs, div_val, mem_len, hybridize):
    batch_size = 8
    vocab_size = 100
    net = TransformerXL(vocab_size=vocab_size, embed_size=embed_size, units=units,
                        embed_cutoffs=cutoffs, embed_div_val=div_val)
    net.initialize()
    if hybridize:
        net.hybridize()

    mems = net.begin_mems(batch_size, mem_len, context=mx.cpu())
    x = mx.nd.arange(batch_size * 16).clip(0, vocab_size - 1).reshape((8, 16))
    y = x
    with mx.autograd.record():
        _ = net(x, y, mems)
    mx.nd.waitall()
