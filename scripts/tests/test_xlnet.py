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
"""Test XLNet."""

import pytest

import mxnet as mx

from ..language_model.transformer import get_model


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.parametrize('use_decoder', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
@pytest.mark.parametrize('modelname', ['xlnet_cased_l12_h768_a12', 'xlnet_cased_l24_h1024_a16'])
def test_xlnet_pretrained(modelname, hybridize, use_decoder):
    model, vocab, tokenizer = get_model(modelname, dataset_name='126gb', use_decoder=use_decoder)
    if hybridize:
        model.hybridize()

    batch_size, qlen, mlen = 2, 16, 100
    mems = model.begin_mems(batch_size, mlen, context=mx.cpu())
    indices = mx.nd.ones(shape=(batch_size, qlen))
    token_types = mx.nd.ones_like(indices)
    output, new_mems = model(indices, token_types, mems)
    mx.nd.waitall()

    assert tokenizer('hello') == ['▁hello']
