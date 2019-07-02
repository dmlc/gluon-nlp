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
import time

import mxnet as mx
import pytest

import gluonnlp as nlp

from gluonnlp.model.transformer import _position_encoding_init

from ..language_model.transformer import AdaptiveEmbedding


@pytest.mark.parametrize('d_head', [5])
@pytest.mark.parametrize('num_heads', [1, 3])
@pytest.mark.parametrize('base_cell', [nlp.model.DotProductAttentionCell(scaled=False, dropout=0)])
@pytest.mark.parametrize('query_length', [10, 5, 1])
@pytest.mark.parametrize('memory_length', [5, 1])
@pytest.mark.parametrize('batch_size', [3, 1])
@pytest.mark.parametrize('use_mask', [True, False])
def test_positional_embedding_multihead_attention_cell(d_head, num_heads, base_cell, query_length,
                                                       memory_length, batch_size, use_mask):
    attention_cell = PositionalEmbeddingMultiHeadAttentionCell(base_cell=base_cell, d_head=d_head,
                                                               num_heads=num_heads)
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


