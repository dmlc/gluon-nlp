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

import sys

import pytest
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import get_model as get_text_model
from common import setup_module, with_seed


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_frequencies(dataset):
    return nlp.data.utils.Counter(x for tup in dataset for x in tup[0]+tup[1][-1:])

@with_seed()
def test_text_models():
    val = nlp.data.WikiText2(segment='val', root='tests/data/wikitext-2')
    val_freq = get_frequencies(val)
    vocab = nlp.Vocab(val_freq)
    text_models = ['standard_lstm_lm_200', 'standard_lstm_lm_650', 'standard_lstm_lm_1500', 'awd_lstm_lm_1150']
    pretrained_to_test = {'standard_lstm_lm_1500': 'wikitext-2', 'standard_lstm_lm_650': 'wikitext-2', 'standard_lstm_lm_200': 'wikitext-2', 'awd_lstm_lm_1150_wikitext-2': 'wikitext-2'}

    for model_name in text_models:
        eprint('testing forward for %s' % model_name)
        pretrained_dataset = pretrained_to_test.get(model_name)
        model, _ = get_text_model(model_name, vocab=vocab, dataset_name=pretrained_dataset,
                                  pretrained=pretrained_dataset is not None, root='tests/data/model/')
        print(model)
        if not pretrained_dataset:
            model.collect_params().initialize()
        output, state = model(mx.nd.arange(330).reshape(33, 10))
        output.wait_to_read()


def test_word_embedding_evaluation_models():
    word_similarity_dataset = nlp.data.WordSim353()

    counter = nlp.data.utils.Counter(
        w for wpair in word_similarity_dataset for w in wpair[:2])
    vocab = nlp.vocab.Vocab(counter)
    vocab.set_embedding(
        nlp.embedding.create("fasttext", source="wiki.simple.vec"))

    data = [[vocab[d[0]], vocab[d[1]], d[2]] for d in word_similarity_dataset]
    words1, words2, scores = zip(*data)

    similarity_evaluator = nlp.model.WordEmbeddingSimilarity(
        vocab_size=len(vocab), embed_size=vocab.embedding.idx_to_vec.shape[1])
    similarity_evaluator.initialize()
    similarity_evaluator.embedding.weight.set_data(vocab.embedding.idx_to_vec)

    pred_similarity = similarity_evaluator(
        mx.nd.array(words1), mx.nd.array(words2))

    sr = nlp.metric.SpearmanRankCorrelation()
    sr.update(mx.nd.array(scores), pred_similarity)

    assert 0.6194264760578906 == pytest.approx(sr.get()[1])


if __name__ == '__main__':
    import nose
    nose.runmodule()
