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

from __future__ import print_function

import pytest
import mxnet as mx
from gluonnlp.model import BiLMEncoder


@pytest.mark.parametrize('hybridize', [False, True])
def test_bilm_encoder_output_shape_lstm(hybridize):
    num_layers = 2
    seq_len = 7
    hidden_size = 100
    input_size = 100
    batch_size = 2

    encoder = BiLMEncoder(mode='lstm',
                          num_layers=num_layers,
                          input_size=input_size,
                          hidden_size=hidden_size,
                          dropout=0.1,
                          skip_connection=False)

    output = run_bi_lm_encoding(encoder, batch_size, input_size, seq_len, hybridize)
    assert output.shape == (num_layers, seq_len, batch_size, 2 * hidden_size), output.shape


@pytest.mark.parametrize('hybridize', [False, True])
def test_bilm_encoder_output_shape_lstmpc(hybridize):
    num_layers = 2
    seq_len = 7
    hidden_size = 100
    input_size = 100
    batch_size = 2
    proj_size = 15

    encoder = BiLMEncoder(mode='lstmpc',
                          num_layers=num_layers,
                          input_size=input_size,
                          hidden_size=hidden_size,
                          dropout=0.1,
                          skip_connection=False,
                          proj_size=proj_size)

    output = run_bi_lm_encoding(encoder, batch_size, input_size, seq_len, hybridize)
    assert output.shape == (num_layers, seq_len, batch_size, 2 * proj_size), output.shape


def run_bi_lm_encoding(encoder, batch_size, input_size, seq_len, hybridize):
    encoder.initialize()

    if hybridize:
        encoder.hybridize()

    inputs = mx.random.uniform(shape=(seq_len, batch_size, input_size))
    inputs_mask = mx.random.uniform(-1, 1, shape=(batch_size, seq_len)) > 1

    state = encoder.begin_state(batch_size=batch_size, func=mx.ndarray.zeros)
    output, _ = encoder(inputs, state, inputs_mask)
    return output
