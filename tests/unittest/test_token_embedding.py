# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import functools
import os

import mxnet as mx
import pytest

import gluonnlp as nlp
from gluonnlp.base import _str_types


class NaiveUnknownLookup:
    def __init__(self, embsize):
        self.embsize = embsize

    def __contains__(self, token):
        return True

    def __getitem__(self, tokens):
        if isinstance(tokens, _str_types):
            return mx.nd.ones(self.embsize)
        else:
            return mx.nd.ones((len(tokens), self.embsize))


@pytest.mark.parametrize('unknown_token', [None, '<unk>', '[UNK]'])
@pytest.mark.parametrize('init_unknown_vec', [mx.nd.zeros, mx.nd.ones])
@pytest.mark.parametrize('allow_extend', [True, False])
@pytest.mark.parametrize('unknown_lookup', [None, NaiveUnknownLookup])
@pytest.mark.parametrize(
    'idx_token_vec_mapping',
    [
        (None, None),
        (['<unk>', 'hello', 'world'], mx.nd.zeros(shape=[3, 300])),  # 300 == embsize
        (['hello', 'world', '<unk>'], mx.nd.zeros(shape=[3, 300])),  # 300 == embsize
        (['hello', 'world'], mx.nd.zeros(shape=[2, 300])),  # 300 == embsize
    ])
def test_token_embedding_constructor(unknown_token, init_unknown_vec, allow_extend, unknown_lookup,
                                     idx_token_vec_mapping, tmp_path, embsize=300):
    idx_to_token, idx_to_vec = idx_token_vec_mapping

    TokenEmbedding = functools.partial(
        nlp.embedding.TokenEmbedding, unknown_token=unknown_token,
        init_unknown_vec=init_unknown_vec, allow_extend=allow_extend,
        unknown_lookup=unknown_lookup(embsize) if unknown_lookup is not None else None,
        idx_to_token=idx_to_token, idx_to_vec=idx_to_vec)

    def test_serialization(emb, tmp_path=tmp_path):
        emb_path = os.path.join(str(tmp_path), "emb.npz")
        emb.serialize(emb_path)
        loaded_emb = nlp.embedding.TokenEmbedding.deserialize(emb_path)
        assert loaded_emb == emb

    ## Test "legacy" constructor
    if idx_to_token is None:
        emb = TokenEmbedding()
        assert len(emb.idx_to_token) == 1 if unknown_token else len(emb.idx_to_token) == 0
        # emb does not know the embsize, thus idx_to_vec could not be initialized
        assert emb.idx_to_vec is None
        with pytest.raises(AttributeError):
            # Cannot serialize as idx_to_vec is not initialized
            test_serialization(emb)

        # Set unknown_token
        if unknown_token:
            emb[unknown_token] = mx.nd.zeros(embsize) - 1
            assert (emb[unknown_token].asnumpy() == mx.nd.zeros(embsize).asnumpy() - 1).all()
            assert emb.idx_to_vec.shape[1] == embsize
            test_serialization(emb)

        if allow_extend:
            emb = TokenEmbedding()
            emb[unknown_token] = mx.nd.zeros(embsize) - 1
            assert emb.idx_to_vec.shape[1] == embsize
            test_serialization(emb)

            emb = TokenEmbedding()
            emb['<some_token>'] = mx.nd.zeros(embsize) - 1
            assert emb.idx_to_vec.shape[0] == 2 if unknown_token else emb.idx_to_vec.shape[0] == 1
            assert (emb['<some_token>'].asnumpy() == (mx.nd.zeros(embsize) - 1).asnumpy()).all()
            test_serialization(emb)

    ## Test with idx_to_vec and idx_to_token arguments
    else:
        emb = TokenEmbedding()

        if unknown_token and unknown_token not in idx_to_token:
            assert emb.idx_to_token == [unknown_token] + idx_to_token
            assert (emb.idx_to_vec[1:].asnumpy() == idx_to_vec.asnumpy()).all()
            assert (emb.idx_to_vec[0].asnumpy() == init_unknown_vec(embsize).asnumpy()).all()
        else:
            assert emb.idx_to_token == idx_to_token
            assert (emb.idx_to_vec.asnumpy() == idx_to_vec.asnumpy()).all()
        test_serialization(emb)

        if allow_extend:
            emb = TokenEmbedding()
            emb[unknown_token] = mx.nd.zeros(embsize) - 1
            assert emb.idx_to_vec.shape[1] == embsize
            test_serialization(emb)

            emb = TokenEmbedding()
            emb['<some_token>'] = mx.nd.zeros(embsize) - 1
            assert (emb['<some_token>'].asnumpy() == (mx.nd.zeros(embsize) - 1).asnumpy()).all()

            if unknown_token and unknown_token not in idx_to_token:
                assert emb.idx_to_vec.shape[0] == len(idx_to_token) + 2
            else:
                assert emb.idx_to_vec.shape[0] == len(idx_to_token) + 1
            test_serialization(emb)
