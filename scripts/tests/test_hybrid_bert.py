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

"""Test hybrid bert models."""

from __future__ import print_function

import os
import sys

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import pytest


from ..bert.export.hybrid_bert import get_hybrid_model


@pytest.mark.serial
@pytest.mark.remote_required
def test_hybrid_bert_models():
    models = ['bert_12_768_12', 'bert_24_1024_16']
    layers = [12, 24]
    attention_heads = [12, 16]
    units = [768, 1024]
    dataset = 'book_corpus_wiki_en_uncased'
    vocab_size = 30522
    batch_size = 2
    seq_len = 3
    num_masks = 2
    ones = mx.nd.ones((batch_size, seq_len))
    valid_length = mx.nd.ones((batch_size,))
    positions = mx.nd.ones((batch_size, num_masks))

    kwargs = [{'use_pooler': False, 'use_decoder': False, 'use_classifier': False},
              {'use_pooler': True, 'use_decoder': False, 'use_classifier': False},
              {'use_pooler': True, 'use_decoder': True, 'use_classifier': False},
              {'use_pooler': True, 'use_decoder': True, 'use_classifier': True},
              {'use_pooler': False, 'use_decoder': False, 'use_classifier': False,
               'output_attention': True},
              {'use_pooler': False, 'use_decoder': False, 'use_classifier': False,
               'output_attention': True, 'output_all_encodings': True},
              {'use_pooler': True, 'use_decoder': True, 'use_classifier': True,
               'output_attention': True, 'output_all_encodings': True}]

    def infer_shape(shapes, unit):
        inferred_shapes = []
        for shape in shapes:
            inferred_shape = list(shape)
            if inferred_shape[-1] == -1:
                inferred_shape[-1] = unit
            inferred_shapes.append(tuple(inferred_shape))
        return inferred_shapes

    def get_shapes(output):
        if not isinstance(output, (list, tuple)):
            return [output.shape]

        shapes = []
        for out in output:
            collect_shapes(out, shapes)

        return shapes

    def collect_shapes(item, shapes):
        if not isinstance(item, (list, tuple)):
            shapes.append(item.shape)
            return

        for child in item:
            collect_shapes(child, shapes)

    for model_name, layer, unit, head in zip(models, layers, units, attention_heads):
        print('testing forward for %s' % model_name)
	
        expected_shapes = [
            [(batch_size, seq_len, -1)],
            [(batch_size, seq_len, -1),
             (batch_size, -1)],
            [(batch_size, seq_len, -1),
             (batch_size, -1),
             (batch_size, num_masks, vocab_size)],
            [(batch_size, seq_len, -1),
             (batch_size, -1),
             (batch_size, 2),
             (batch_size, num_masks, vocab_size)],
            [(batch_size, seq_len, -1)] + [(num_masks, head, seq_len, seq_len)] * layer,
            [(batch_size, seq_len, -1)] * layer + [(num_masks, head, seq_len, seq_len)] * layer,
            [(batch_size, seq_len, -1)] * layer + [(num_masks, head, seq_len, seq_len)] * layer +
            [(batch_size, -1)] + [(batch_size, 2)] + [(batch_size, num_masks, vocab_size)],
        ]

        for kwarg, expected_shape in zip(kwargs, expected_shapes):
            expected_shape = infer_shape(expected_shape, unit)
            model, _ = get_hybrid_model(model_name, dataset_name=dataset,
                                        pretrained=False, root='tests/data/model/',
                                        seq_length=seq_len, input_size=unit,
                                        **kwarg)
            model.initialize()
            if kwarg['use_decoder']:
                # position tensor is required for decoding
                output = model(ones, ones, valid_length, positions)
            else:
                output = model(ones, ones, valid_length)
            out_shapes = get_shapes(output)
            assert out_shapes == expected_shape, (out_shapes, expected_shape)
            sync_instance = output[0] if not isinstance(output[0], list) else output[0][0]
            sync_instance.wait_to_read()
            del model
            mx.nd.waitall()

