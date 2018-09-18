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
from mxnet import gluon
import gluonnlp as nlp
import pytest

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# disabled since it takes a long time to download the model
@pytest.mark.serial
def _test_pretrained_big_text_models():
    text_models = ['big_rnn_lm_2048_512']
    pretrained_to_test = {'big_rnn_lm_2048_512': 'gbw'}

    for model_name in text_models:
        eprint('testing forward for %s' % model_name)
        pretrained_dataset = pretrained_to_test.get(model_name)
        model, _ = nlp.model.get_model(model_name, dataset_name=pretrained_dataset,
                                       pretrained=True, root='tests/data/model/')

        print(model)
        batch_size = 10
        hidden = model.begin_state(batch_size=batch_size, func=mx.nd.zeros)
        output, state = model(mx.nd.arange(330).reshape((33, 10)), hidden)
        output.wait_to_read()

@pytest.mark.serial
def test_big_text_models(wikitext2_val_and_counter):
    # use a small vocabulary for testing
    val, val_freq = wikitext2_val_and_counter
    vocab = nlp.Vocab(val_freq)
    text_models = ['big_rnn_lm_2048_512']

    for model_name in text_models:
        eprint('testing forward for %s' % model_name)
        model, _ = nlp.model.get_model(model_name, vocab=vocab, root='tests/data/model/')

        print(model)
        model.collect_params().initialize()
        batch_size = 10
        hidden = model.begin_state(batch_size=batch_size, func=mx.nd.zeros)
        output, state = model(mx.nd.arange(330).reshape((33, 10)), hidden)
        output.wait_to_read()

@pytest.mark.serial
def test_text_models():
    text_models = ['standard_lstm_lm_200', 'standard_lstm_lm_650', 'standard_lstm_lm_1500', 'awd_lstm_lm_1150', 'awd_lstm_lm_600']
    pretrained_to_test = {'standard_lstm_lm_1500': 'wikitext-2',
                          'standard_lstm_lm_650': 'wikitext-2',
                          'standard_lstm_lm_200': 'wikitext-2',
                          'awd_lstm_lm_1150': 'wikitext-2',
                          'awd_lstm_lm_600': 'wikitext-2'}

    for model_name in text_models:
        eprint('testing forward for %s' % model_name)
        pretrained_dataset = pretrained_to_test.get(model_name)
        model, _ = nlp.model.get_model(model_name, dataset_name=pretrained_dataset,
                                       pretrained=pretrained_dataset is not None,
                                       root='tests/data/model/')

        print(model)
        if not pretrained_dataset:
            model.collect_params().initialize()
        output, state = model(mx.nd.arange(330).reshape(33, 10))
        output.wait_to_read()
        del model
        mx.nd.waitall()

@pytest.mark.serial
def test_cache_models():
    cache_language_models = ['awd_lstm_lm_1150', 'awd_lstm_lm_600', 'standard_lstm_lm_200',
                   'standard_lstm_lm_650', 'standard_lstm_lm_1500']
    datasets = ['wikitext-2']
    for name in cache_language_models:
        for dataset_name in datasets:
            cache_cell = nlp.model.train.get_cache_model(name, dataset_name, window=1, theta=0.6,
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


@pytest.mark.serial
def test_get_cache_model_noncache_models():
    language_models_params = {'awd_lstm_lm_1150': 'awd_lstm_lm_1150_wikitext-2-f9562ed0.params',
                              'awd_lstm_lm_600': 'awd_lstm_lm_600_wikitext-2-e952becc.params',
                              'standard_lstm_lm_200': 'standard_lstm_lm_200_wikitext-2-b233c700.params',
                              'standard_lstm_lm_650': 'standard_lstm_lm_650_wikitext-2-631f3904.params',
                              'standard_lstm_lm_1500': 'standard_lstm_lm_1500_wikitext-2-a4163513.params'}
    datasets = ['wikitext-2']
    for name in language_models_params.keys():
        for dataset_name in datasets:
            _, vocab = nlp.model.get_model(name=name, dataset_name=dataset_name, pretrained=True,
                                           root='tests/data/model')
            ntokens = len(vocab)

            cache_cell_0 = nlp.model.train.get_cache_model(name, dataset_name, window=1, theta=0.6,
                                                           lambdas=0.2, root='tests/data/model/')
            print(cache_cell_0)

            model, _ = nlp.model.get_model(name=name, dataset_name=dataset_name, pretrained=True,
                                           root='tests/data/model/')
            cache_cell_1 = nlp.model.train.CacheCell(model, ntokens, window=1, theta=0.6, lambdas=0.2)
            cache_cell_1.load_parameters('tests/data/model/' + language_models_params.get(name))
            print(cache_cell_1)

            outs0, word_history0, cache_history0, hidden0 = \
                cache_cell_0(mx.nd.arange(10).reshape(10, 1), mx.nd.arange(10).reshape(10, 1), None, None)
            outs1, word_history1, cache_history1, hidden1 = \
                cache_cell_1(mx.nd.arange(10).reshape(10, 1), mx.nd.arange(10).reshape(10, 1), None, None)

            assert outs0.shape == outs1.shape, outs0.shape
            assert len(word_history0) == len(word_history1), len(word_history0)
            assert len(cache_history0) == len(cache_history1), len(cache_history0)
            assert len(hidden0) == len(hidden1), len(hidden0)


@pytest.mark.serial
def test_save_load_cache_models():
    cache_language_models = ['awd_lstm_lm_1150', 'awd_lstm_lm_600', 'standard_lstm_lm_200',
                   'standard_lstm_lm_650', 'standard_lstm_lm_1500']
    datasets = ['wikitext-2']
    for name in cache_language_models:
        for dataset_name in datasets:
            cache_cell = nlp.model.train.get_cache_model(name, dataset_name, window=1, theta=0.6,
                                                         lambdas=0.2, root='tests/data/model/')
            print(cache_cell)
            cache_cell.save_parameters('tests/data/model/' + name + '-' + dataset_name + '.params')
            cache_cell.load_parameters('tests/data/model/' + name + '-' + dataset_name + '.params')

@pytest.mark.serial
def test_save_load_big_rnn_models():
    ctx = mx.cpu()
    seq_len = 1
    batch_size = 1
    num_sampled = 6
    # network
    eval_model = nlp.model.language_model.BigRNN(10, 2, 3, 4, 5, 0.1, prefix='bigrnn')
    model = nlp.model.language_model.train.BigRNN(10, 2, 3, 4, 5, num_sampled, 0.1,
                                                  prefix='bigrnn')
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    # verify param names
    model_params = sorted(model.collect_params().keys())
    eval_model_params = sorted(eval_model.collect_params().keys())
    for p0, p1 in zip(model_params, eval_model_params):
        assert p0 == p1, (p0, p1)
    model.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = mx.gluon.Trainer(model.collect_params(), 'sgd')
    # prepare data, label and samples
    x = mx.nd.ones((seq_len, batch_size))
    y = mx.nd.ones((seq_len, batch_size))
    sampled_cls = mx.nd.ones((num_sampled,))
    sampled_cls_cnt = mx.nd.ones((num_sampled,))
    true_cls_cnt = mx.nd.ones((seq_len,batch_size))
    samples = (sampled_cls, sampled_cls_cnt, true_cls_cnt)
    hidden = model.begin_state(batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    # test forward
    with mx.autograd.record():
        pred, hidden, new_y = model(x, y, hidden, samples)
        assert pred.shape == (seq_len, batch_size, 1+num_sampled)
        assert new_y.shape == (seq_len, batch_size)
        pred = pred.reshape((-3, -1))
        new_y = new_y.reshape((-1,))
        l = loss(pred, new_y)
    l.backward()
    mx.nd.waitall()
    path = 'tests/data/model/test_save_load_big_rnn_models.params'
    model.save_parameters(path)
    eval_model.load_parameters(path)

def test_big_rnn_model_share_params():
    ctx = mx.cpu()
    seq_len = 2
    batch_size = 1
    num_sampled = 6
    vocab_size = 10
    shape = (seq_len, batch_size)
    model = nlp.model.language_model.train.BigRNN(vocab_size, 2, 3, 4, 5, num_sampled, 0.1,
                                                  prefix='bigrnn', sparse_weight=False,
                                                  sparse_grad=False)
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    model.hybridize()
    model.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = mx.gluon.Trainer(model.collect_params(), 'sgd')
    batch_size = 1
    x = mx.nd.ones(shape)
    y = mx.nd.ones(shape)
    sampled_cls = mx.nd.ones((num_sampled,))
    sampled_cls_cnt = mx.nd.ones((num_sampled,))
    true_cls_cnt = mx.nd.ones(shape)
    samples = (sampled_cls, sampled_cls_cnt, true_cls_cnt)
    hidden = model.begin_state(batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    with mx.autograd.record():
        pred, hidden, new_y = model(x, y, hidden, samples)
        assert pred.shape == (seq_len, batch_size, 1+num_sampled)
        assert new_y.shape == (seq_len, batch_size)
        pred = pred.reshape((-3, -1))
        new_y = new_y.reshape((-1,))
        l = loss(pred, new_y)
    l.backward()
    assert model.decoder.weight._grad_stype == 'default'
    mx.nd.waitall()
    eval_model = nlp.model.language_model.BigRNN(vocab_size, 2, 3, 4, 5, 0.1, prefix='bigrnn',
                                                 params=model.collect_params())
    eval_model.hybridize()
    eval_model.initialize(mx.init.Xavier(), ctx=ctx)
    pred, hidden = eval_model(x, hidden)
    assert pred.shape == (seq_len, batch_size, vocab_size)
    mx.nd.waitall()

def test_weight_drop():
    class RefBiLSTM(gluon.Block):
        def __init__(self, size, **kwargs):
            super(RefBiLSTM, self).__init__(**kwargs)
            with self.name_scope():
                self._lstm_fwd = gluon.rnn.LSTM(size, bidirectional=False, prefix='l0')
                self._lstm_bwd = gluon.rnn.LSTM(size, bidirectional=False, prefix='r0')

        def forward(self, inpt):
            fwd = self._lstm_fwd(inpt)
            bwd_inpt = mx.nd.flip(inpt, 0)
            bwd = self._lstm_bwd(bwd_inpt)
            bwd = mx.nd.flip(bwd, 0)
            return mx.nd.concat(fwd, bwd, dim=2)
    net1 = RefBiLSTM(10)
    shared_net1 = RefBiLSTM(10, params=net1.collect_params())

    net2 = gluon.rnn.LSTM(10)
    shared_net2 = gluon.rnn.LSTM(10, params=net2.collect_params())

    net3 = gluon.nn.HybridSequential()
    net3.add(gluon.rnn.LSTM(10))
    shared_net3 = gluon.nn.HybridSequential(params=net3.collect_params())
    shared_net3.add(gluon.rnn.LSTM(10, params=net3[0].collect_params()))

    x = mx.nd.ones((3, 4, 5))
    nets = [(net1, shared_net1),
            (net2, shared_net2),
            (net3, shared_net3)]
    for net, shared_net in nets:
        net.initialize('ones')
        mx.test_utils.assert_almost_equal(net(x).asnumpy(),
                                          shared_net(x).asnumpy())
        with mx.autograd.train_mode():
            mx.test_utils.assert_almost_equal(net(x).asnumpy(),
                                              shared_net(x).asnumpy())

        grads = {}
        with mx.autograd.record():
            y = net(x)
        y.backward()
        for name, param in net.collect_params().items():
            grads[name] = param.grad().copy()
        with mx.autograd.record():
            y = shared_net(x)
        y.backward()
        for name, param in shared_net.collect_params().items():
            mx.test_utils.assert_almost_equal(grads[name].asnumpy(), param.grad().asnumpy())

        drop_rate = 0.5
        nlp.model.utils.apply_weight_drop(net, '.*h2h_weight', drop_rate)
        net.initialize('ones')

        mx.test_utils.assert_almost_equal(net(x).asnumpy(),
                                          shared_net(x).asnumpy())
        with mx.autograd.train_mode():
            assert not mx.test_utils.almost_equal(net(x).asnumpy(),
                                                  shared_net(x).asnumpy())

        grads = {}
        with mx.autograd.record():
            y = net(x)
        y.backward()
        for name, param in net.collect_params().items():
            grads[name] = param.grad().copy()
        with mx.autograd.record():
            y = shared_net(x)
        y.backward()
        for name, param in shared_net.collect_params().items():
            assert not mx.test_utils.almost_equal(grads[name].asnumpy(), param.grad().asnumpy())
