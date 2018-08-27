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

import os

import pytest

import gluonnlp as nlp
import mxnet as mx


@pytest.mark.parametrize('batch_size', [7, 80])
@pytest.mark.serial
def test_corpus_batchify(batch_size, wikitext2_test_and_counter):
    data, counter = wikitext2_test_and_counter
    vocab = nlp.Vocab(counter)
    batchify = nlp.data.batchify.CorpusBatchify(vocab, batch_size)
    batches = batchify(data)
    assert batches[:].shape == (len(data) // batch_size, batch_size)


@pytest.mark.parametrize('batch_size', [7, 80])
@pytest.mark.parametrize('seq_len', [7, 35])
@pytest.mark.serial
def test_corpus_bptt_batchify(batch_size, seq_len, wikitext2_test_and_counter):
    data, counter = wikitext2_test_and_counter
    vocab = nlp.Vocab(counter)

    # unsupported last_batch
    with pytest.raises(ValueError):
        bptt_keep = nlp.data.batchify.CorpusBPTTBatchify(
            vocab, seq_len, batch_size, last_batch='unsupported')

    # last_batch='keep'
    bptt_keep = nlp.data.batchify.CorpusBPTTBatchify(
        vocab, seq_len, batch_size, last_batch='keep')
    X, Y = zip(*(bptt_keep(data)))
    X, Y = mx.nd.concat(*X, dim=0), mx.nd.concat(*Y, dim=0)
    coded = mx.nd.concat(
        X, Y[-1].expand_dims(0), dim=0).T.reshape(-1).asnumpy().tolist()
    assert vocab[list(data)] == coded[:len(data)]
    assert all(pad == vocab[vocab.padding_token] for pad in coded[len(data):])

    # last_batch='discard'
    bptt_discard = nlp.data.batchify.CorpusBPTTBatchify(
        vocab, seq_len, batch_size, last_batch='discard')
    X, Y = zip(*(bptt_discard(data)))
    X, Y = mx.nd.concat(*X, dim=0), mx.nd.concat(*Y, dim=0)
    coded = mx.nd.concat(
        X, Y[-1].expand_dims(0), dim=0).T.reshape(-1).asnumpy().tolist()
    assert len(data) - len(coded) < batch_size * seq_len


@pytest.mark.serial
def test_bptt_batchify_padding_token():
    vocab = nlp.Vocab(
        nlp.data.utils.Counter(['a', 'b', 'c']), padding_token=None)
    seq_len = 35
    batch_size = 80

    # Padding token must always be specified for StreamBPTTBatchify
    with pytest.raises(ValueError):
        nlp.data.batchify.StreamBPTTBatchify(
            vocab, seq_len, batch_size, last_batch='discard')

    with pytest.raises(ValueError):
        nlp.data.batchify.StreamBPTTBatchify(
            vocab, seq_len, batch_size, last_batch='keep')

    # Padding token must be specified for last_batch='keep' for CorpusBPTTBatchify
    with pytest.raises(ValueError):
        nlp.data.batchify.CorpusBPTTBatchify(
            vocab, seq_len, batch_size, last_batch='keep')

    nlp.data.batchify.CorpusBPTTBatchify(
        vocab, seq_len, batch_size, last_batch='discard')


@pytest.mark.parametrize('batch_size', [7, 80])
@pytest.mark.parametrize('seq_len', [7, 35])
@pytest.mark.serial
def test_stream_bptt_batchify(
        seq_len, batch_size, stream_identity_wrappers,
        wikitext2_simpledatasetstream_skipempty_and_counter):
    stream, counter = wikitext2_simpledatasetstream_skipempty_and_counter
    vocab = nlp.vocab.Vocab(counter, bos_token=None)

    bptt_keep = nlp.data.batchify.StreamBPTTBatchify(
        vocab, seq_len, batch_size, last_batch='keep')
    bptt_stream = stream_identity_wrappers(bptt_keep(stream))
    padding_idx = vocab[vocab.padding_token]
    total_num_tokens = sum(counter.values())
    num_tokens_per_batch = seq_len * batch_size
    num_tokens = 0
    for i, (data, target) in enumerate(bptt_stream):
        mask = data != padding_idx
        # count the valid tokens in the batch
        num_valid_tokens = mask.sum().asscalar()
        if num_valid_tokens == num_tokens_per_batch:
            mx.test_utils.assert_almost_equal(data[1:].asnumpy(), target[:-1].asnumpy())
            assert data.shape == target.shape == (seq_len, batch_size)
        num_tokens += num_valid_tokens
    num_batches = sum(1 for _ in bptt_stream)
    # the last token doesn't appear in data
    assert num_tokens < total_num_tokens
