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


def test_elmo_bilm_encoder():
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
    mask = mx.nd.ones(shape=(5, 20))
    states = encoder.begin_state(mx.nd.zeros, batch_size=5)
    print('testing forward for %s' % 'elmo bilm')
    outputs, out_states = encoder(inputs, states, mask)

    assert outputs.shape == (1, 20, 5, 20), outputs.shape
    assert len(out_states) == 2, len(out_states)
    assert out_states[0][0][0].shape == (5, 10), out_states[0][0][0].shape
    assert out_states[0][0][1].shape == (5, 30), out_states[0][0][1].shape
    assert out_states[1][0][0].shape == (5, 10), out_states[0][1][0].shape
    assert out_states[1][0][1].shape == (5, 30), out_states[0][1][1].shape


def test_elmo_char_encoder():
    char_encoder = nlp.model.ELMoCharacterEncoder(output_size=1,
                                                  char_embed_size=2,
                                                  filters=[[1, 2], [2, 1]],
                                                  num_highway=2,
                                                  conv_layer_activation='relu',
                                                  max_chars_per_token=50)
    print(char_encoder)
    char_encoder.initialize()
    inputs = mx.nd.ones(shape=(2, 5, 50))
    print('testing forward for %s' % 'elmo_char_encoder')
    mask, output = char_encoder(inputs)
    assert mask.shape == (2, 7), mask.shape
    assert output.shape == (2, 7, 1), output.shape


def test_elmo_model():
    model = nlp.model.ELMoBiLM(model='lstmpc',
                     output_size=128,
                     filters=[[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
                     char_embed_size=16,
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
    inputs = mx.nd.ones(shape=(5, 20, 50))
    begin_state = model.begin_state(mx.nd.zeros, batch_size=5)
    print('testing forward for %s' % 'elmo model')
    outputs, state, mask = model(inputs, begin_state)
    assert len(outputs) == 3, len(outputs)
    assert outputs[0].shape == (5, 22, 256), outputs[0].shape
    assert len(state) == 2, len(state)
    assert mask.shape == (5, 22), mask.shape


@pytest.mark.serial
@pytest.mark.remote_required
def test_get_elmo_models():
    model_names = ['elmo_2x1024_128_2048cnn_1xhighway', 'elmo_2x2048_256_2048cnn_1xhighway',
                   'elmo_2x4096_512_2048cnn_2xhighway', 'elmo_2x4096_512_2048cnn_2xhighway']
    datasets = ['gbw', 'gbw', 'gbw', '5bw']

    for model_name, dataset in zip(model_names, datasets):
        print('testing forward for %s on dataset %s' % (model_name, dataset))
        model = nlp.model.get_model(model_name,
                                    dataset_name=dataset,
                                    pretrained=dataset is not None,
                                    root='tests/data/model/')

        print(model)
        if not dataset:
            model.collect_params().initialize()
        begin_state = model.begin_state(mx.nd.zeros, batch_size=20)
        output, state, mask = model(mx.nd.arange(35000).reshape(20, 35, 50), begin_state)
        del model
        mx.nd.waitall()
