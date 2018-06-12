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

from gluonnlp.model import CharacterEncoder

def test_char_encoder_nonhighway_forward():
    encoder = CharacterEncoder(embed_size=2, num_filters=1, ngram_filter_sizes=(1,2))
    encoder.initialize()
    input = mx.nd.array([[[.7, .8], [.1, 1.5]]])
    output = encoder(input, None)
    assert output.equals(mx.nd.array([[1.6 + 1.0, 3.1 + 1.0]])), output

def test_char_encoder_highway_forward():
    encoder = CharacterEncoder(embed_size=2, num_filters=1, ngram_filter_sizes=(1,2), num_highway=1)
    encoder.initialize()
    input = mx.nd.array([[[.7, .8], [.1, 1.5]]])
    output = encoder(input, None)
    assert output.equals(mx.nd.array([[1.6 + 1.0, 3.1 + 1.0]])), output

test_char_encoder_nonhighway_forward()