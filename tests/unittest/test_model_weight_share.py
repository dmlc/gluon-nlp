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

import warnings

import mxnet as mx
import gluonnlp as nlp
import pytest

def _check_initialized(net):
    params = net.collect_params()
    for param in params:
        try:
            params[param].list_ctx()
        except RuntimeError:
            return False
        return True

@pytest.mark.parametrize('weight_tied', [False, True])
def test_awdrnn_weight_share(weight_tied):
    mode = 'lstm'
    vocab = 400
    context = [mx.cpu()]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = nlp.model.train.AWDRNN(mode, vocab,
                                       tie_weights=weight_tied)
        model_eval = nlp.model.train.AWDRNN(mode, vocab,
                                            tie_weights=weight_tied,
                                            params=model.collect_params())
        model.initialize(mx.init.Xavier(), ctx=context)

        assert _check_initialized(model) == True
        assert _check_initialized(model_eval) == True

@pytest.mark.parametrize('weight_tied', [False, True])
def test_standardrnn_weight_share(weight_tied):
    mode = 'lstm'
    vocab = 400
    context = [mx.cpu()]
    emb_size = 200
    hidden_size = 200
    nlayers = 2

    model = nlp.model.train.StandardRNN(mode, vocab,
                                        emb_size, hidden_size,
                                        nlayers, weight_tied)
    model_eval = nlp.model.train.StandardRNN(mode, vocab,
                                             emb_size, hidden_size,
                                             nlayers, weight_tied,
                                             params=model.collect_params())
    model.initialize(mx.init.Xavier(), ctx=context)

    assert _check_initialized(model) == True
    assert _check_initialized(model_eval) == True
