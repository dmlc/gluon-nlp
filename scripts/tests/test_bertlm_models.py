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
import pytest

from ..language_model.transformer_lm import bert_lm_12_768_12_300_1150, \
    bert_lm_12_768_12_400_2500, bert_lm_24_1024_16_300_1150, bert_lm_24_1024_16_400_2500


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@pytest.mark.serial
@pytest.mark.remote_required
def test_bert_language_models():
    models = [bert_lm_12_768_12_300_1150, bert_lm_12_768_12_400_2500,
              bert_lm_24_1024_16_300_1150, bert_lm_24_1024_16_400_2500]
    model_names = ['bert_lm_12_768_12_300_1150', 'bert_lm_12_768_12_400_2500',
                   'bert_lm_24_1024_16_300_1150', 'bert_lm_24_1024_16_400_2500']
    datasets = ['wikitext2', 'wikitext103', 'wikitext2', 'wikitext103']

    for i, model_name in enumerate(model_names):
        eprint('testing forward for %s' % model_name)
        pretrained_dataset = datasets[i]
        model, _ = models[i](pretrained_dataset,
                             pretrained=pretrained_dataset is not None,
                             root='tests/data/model/')

        print(model)
        if not pretrained_dataset:
            model.collect_params().initialize()
        inputs = mx.nd.arange(330).reshape(33, 10)
        output, state = model(inputs)
        output.wait_to_read()

        assert output.shape == (33, 10, 30522), output.shape
        del model
        mx.nd.waitall()
