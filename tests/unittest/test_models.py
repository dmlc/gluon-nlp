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

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import get_model as get_text_model
from gluonnlp.model.train import get_cache_model
from gluonnlp.model.train import CacheCell

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_frequencies(dataset):
    return nlp.data.utils.Counter(x for tup in dataset for x in tup[0]+tup[1][-1:])

def test_text_models():
    val = nlp.data.WikiText2(segment='val', root='tests/data/wikitext-2')
    val_freq = get_frequencies(val)
    vocab = nlp.Vocab(val_freq)
    text_models = ['standard_lstm_lm_200', 'standard_lstm_lm_650', 'standard_lstm_lm_1500', 'awd_lstm_lm_1150', 'awd_lstm_lm_600']
    pretrained_to_test = {'standard_lstm_lm_1500': 'wikitext-2', 'standard_lstm_lm_650': 'wikitext-2', 'standard_lstm_lm_200': 'wikitext-2', 'awd_lstm_lm_1150': 'wikitext-2', 'awd_lstm_lm_600': 'wikitext-2'}

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

def test_cache_models():
    cache_language_models = ['awd_lstm_lm_1150', 'awd_lstm_lm_600', 'standard_lstm_lm_200',
                   'standard_lstm_lm_650', 'standard_lstm_lm_1500']
    datasets = ['wikitext-2']
    for name in cache_language_models:
        for dataset_name in datasets:
            cache_cell = get_cache_model(name, dataset_name, window=1, theta=0.6,
                                         lambdas=0.2, root='tests/data/model/')
            outs, word_history, cache_history, hidden = \
                cache_cell(mx.nd.arange(10).reshape(10, 1), mx.nd.arange(10).reshape(10, 1), None, None)
            print(cache_cell)
            print("outs:")
            print(outs)
            print("word_history:")
            print(word_history)
            print("cache_history:")
            print(cache_history)


def test_get_cache_model_noncache_models():
    language_models_params = {'awd_lstm_lm_1150': 'awd_lstm_lm_1150_wikitext-2-45d6df33.params',
                              'awd_lstm_lm_600': 'awd_lstm_lm_600_wikitext-2-7894a046.params',
                              'standard_lstm_lm_200': 'standard_lstm_lm_200_wikitext-2-700b532d.params',
                              'standard_lstm_lm_650': 'standard_lstm_lm_650_wikitext-2-14041667.params',
                              'standard_lstm_lm_1500': 'standard_lstm_lm_1500_wikitext-2-d572ce71.params'}
    datasets = ['wikitext-2']
    for name in language_models_params.keys():
        for dataset_name in datasets:
            _, vocab = get_text_model(name=name, dataset_name=dataset_name, pretrained=True)
            ntokens = len(vocab)

            cache_cell_0 = get_cache_model(name, dataset_name, window=1, theta=0.6,
                                           lambdas=0.2, root='tests/data/model/')
            print(cache_cell_0)

            model, _ = get_text_model(name=name, dataset_name=dataset_name, pretrained=True)
            cache_cell_1 = CacheCell(model, ntokens, window=1, theta=0.6, lambdas=0.2)
            cache_cell_1.load_params('tests/data/model/' + language_models_params.get(name))
            print(cache_cell_1)

            outs0, word_history0, cache_history0, hidden0 = \
                cache_cell_0(mx.nd.arange(10).reshape(10, 1), mx.nd.arange(10).reshape(10, 1), None, None)
            outs1, word_history1, cache_history1, hidden1 = \
                cache_cell_1(mx.nd.arange(10).reshape(10, 1), mx.nd.arange(10).reshape(10, 1), None, None)

            assert outs0.shape == outs1.shape, outs0.shape
            assert len(word_history0) == len(word_history1), len(word_history0)
            assert len(cache_history0) == len(cache_history1), len(cache_history0)
            assert len(hidden0) == len(hidden1), len(hidden0)


def test_save_load_cache_models():
    cache_language_models = ['awd_lstm_lm_1150', 'awd_lstm_lm_600', 'standard_lstm_lm_200',
                   'standard_lstm_lm_650', 'standard_lstm_lm_1500']
    datasets = ['wikitext-2']
    for name in cache_language_models:
        for dataset_name in datasets:
            cache_cell = get_cache_model(name, dataset_name, window=1, theta=0.6,
                                           lambdas=0.2, root='tests/data/model/')
            print(cache_cell)
            cache_cell.save_params('tests/data/model/' + name + '-' + dataset_name + '.params')
            cache_cell.load_params('tests/data/model/' + name + '-' + dataset_name + '.params')
