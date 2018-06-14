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

import mxnet as mx
from numpy.testing import assert_almost_equal

from gluonnlp.model import CharacterEncoder


def test_char_encoder_nonhighway_forward():
    encoder = CharacterEncoder(embed_size=2, num_filters=1, ngram_filter_sizes=(1, 2))
    print(encoder)
    encoder.initialize(init='One')
    inputs = mx.nd.array([[[.7, .8], [.1, 1.5], [.2, .3]], [[.5, .6], [.2, 2.5], [.4, 4]]])
    output = encoder(inputs, None)
    assert output.shape == (3, 2), output.shape
    assert_almost_equal(output.asnumpy(),
                        mx.nd.array([[1.5, 2.6], [2.7, 4.3], [4.4, 4.9]]).asnumpy(),
                        decimal=2)


def test_char_encoder_nohighway_forward_largeinputs():
    encoder = CharacterEncoder(embed_size=7,
                               num_filters=13,
                               ngram_filter_sizes=(1, 2, 3, 4),
                               output_size=30)
    print(encoder)
    encoder.initialize()
    inputs = mx.nd.random.uniform(shape=(4, 8, 7))
    output = encoder(inputs, None)
    assert output.shape == (8, 30), output.shape


def test_char_encoder_highway_forward():
    encoder = CharacterEncoder(embed_size=2, num_filters=1, ngram_filter_sizes=(1, 2), num_highway=2, output_size=1)
    print(encoder)
    encoder.initialize()
    encoder.set_highway_bias()
    inputs = mx.nd.array([[[.7, .8], [.1, 1.5], [.7, .8]], [[.7, .8], [.1, 1.5], [.7, .8]]])
    output = encoder(inputs, None)
    print(output)
    assert output.shape == (3, 1), output.shape
