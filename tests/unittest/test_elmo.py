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


import mxnet as mx
import gluonnlp as nlp
import pytest


@pytest.mark.parametrize('has_mask', [False, True])
def test_elmo_bilm_encoder(has_mask):
    encoder = nlp.model.BiLMEncoder(mode='lstmpc',
                                    num_layers=1,
                                    input_size=10,
                                    hidden_size=30,
                                    dropout=0.1,
                                    skip_connection=True,
                                    proj_size=10,
                                    cell_clip=1,
                                    proj_clip=1)
    print(encoder)
    encoder.initialize()
    inputs = mx.random.uniform(shape=(20, 5, 10))
    states = encoder.begin_state(mx.nd.zeros, batch_size=5)
    if has_mask:
        mask = mx.nd.ones(shape=(5, 20))
        print('testing forward for elmo bilm with mask')
        outputs, out_states = encoder(inputs, states, mask)
    else:
        print('testing forward for elmo bilm without mask')
        outputs, out_states = encoder(inputs, states)

    assert outputs.shape == (1, 20, 5, 20), outputs.shape
    assert len(out_states) == 2, len(out_states)
    assert out_states[0][0][0].shape == (5, 10), out_states[0][0][0].shape
    assert out_states[0][0][1].shape == (5, 30), out_states[0][0][1].shape
    assert out_states[1][0][0].shape == (5, 10), out_states[0][1][0].shape
    assert out_states[1][0][1].shape == (5, 30), out_states[0][1][1].shape


@pytest.mark.parametrize('hybridize', [False, True])
def test_elmo_char_encoder(hybridize):
    char_encoder = nlp.model.ELMoCharacterEncoder(output_size=1,
                                                  char_embed_size=2,
                                                  filters=[[1, 2], [2, 1]],
                                                  num_highway=2,
                                                  conv_layer_activation='relu',
                                                  max_chars_per_token=50,
                                                  char_vocab_size=262)
    print(char_encoder)
    char_encoder.initialize()
    if hybridize:
        char_encoder.hybridize()
    inputs = mx.nd.ones(shape=(2, 5, 50))
    print('testing forward for %s' % 'elmo_char_encoder')
    output = char_encoder(inputs)
    assert output.shape == (2, 5, 1), output.shape


@pytest.mark.parametrize('hybridize', [False, True])
def test_elmo_model(hybridize):
    filters=[[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
    model = nlp.model.ELMoBiLM(rnn_type='lstmpc',
                               output_size=128,
                               filters=filters,
                               char_embed_size=16,
                               char_vocab_size=262,
                               num_highway=1,
                               conv_layer_activation='relu',
                               max_chars_per_token=50,
                               input_size=128,
                               hidden_size=1024,
                               proj_size=128,
                               num_layers=2,
                               cell_clip=1,
                               proj_clip=1,
                               skip_connection=True)
    print(model)
    model.initialize()
    if hybridize:
        model.hybridize()
    inputs = mx.nd.ones(shape=(5, 20, 50))
    begin_state = model.begin_state(mx.nd.zeros, batch_size=5)
    print('testing forward for %s' % 'elmo model')
    outputs, state = model(inputs, begin_state)
    assert len(outputs) == 3, len(outputs)
    assert outputs[0].shape == (5, 20, 256), outputs[0].shape
    assert len(state) == 2, len(state)


@pytest.mark.serial
@pytest.mark.remote_required
def test_get_elmo_models():
    model_names = ['elmo_2x1024_128_2048cnn_1xhighway', 'elmo_2x2048_256_2048cnn_1xhighway',
                   'elmo_2x4096_512_2048cnn_2xhighway', 'elmo_2x4096_512_2048cnn_2xhighway']
    datasets = ['gbw', 'gbw', 'gbw', '5bw']

    for model_name, dataset in zip(model_names, datasets):
        print('testing forward for %s on dataset %s' % (model_name, dataset))
        model, _ = nlp.model.get_model(model_name,
                                       dataset_name=dataset,
                                       pretrained=dataset is not None,
                                       root='tests/data/model/')

        print(model)
        if not dataset:
            model.collect_params().initialize()
        begin_state = model.begin_state(mx.nd.zeros, batch_size=20)
        output, state = model(mx.nd.arange(35000).reshape(20, 35, 50), begin_state)
        del model
        mx.nd.waitall()

def test_elmo_vocab():
    vocab = nlp.vocab.ELMoCharVocab()
    expected_bos_ids = [vocab.bow_id, vocab.bos_id, vocab.eow_id]+[vocab.pad_id]*(vocab.max_word_length-3)
    expected_eos_ids = [vocab.bow_id, vocab.eos_id, vocab.eow_id]+[vocab.pad_id]*(vocab.max_word_length-3)
    expected_hello_ids = [vocab.bow_id, 104, 101, 108, 108, 111, vocab.eow_id]+[vocab.pad_id]*(vocab.max_word_length-7)
    assert vocab['<bos>'] == expected_bos_ids
    assert vocab['<eos>'] == expected_eos_ids
    assert vocab['hello'] == expected_hello_ids
    assert vocab[['<bos>', 'hello', '<eos>']] == [expected_bos_ids, expected_hello_ids, expected_eos_ids]
