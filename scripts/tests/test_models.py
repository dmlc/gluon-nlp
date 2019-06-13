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

"""Test models that are not in API yet."""

import pytest
import numpy as np
import numpy.testing as npt

import mxnet as mx
from mxnet.gluon.utils import _get_repo_url, download

from gluonnlp.data.transforms import GPT2BPETokenizer, GPT2BPEDetokenizer
from ..text_generation.model import get_model

@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', ['gpt2_117m', 'gpt2_345m'])
def test_pretrained_gpt2(model_name):
    sentence = ' natural language processing tools such as gluonnlp and torchtext'
    model, vocab = get_model(model_name, dataset_name='openai_webtext')
    tokenizer = GPT2BPETokenizer()
    detokenizer = GPT2BPEDetokenizer()
    true_data_hash = {'gpt2_117m': '29526682508d03a7c54c598e889f77f7b4608df0',
                      'gpt2_345m': '6680fd2a3d7b737855536f480bc19d166f15a3ad'}
    file_name = '{model_name}_gt_logits-{short_hash}.npy'.format(
            model_name=model_name,
            short_hash=true_data_hash[model_name][:8])
    url_format = '{repo_url}gluon/dataset/test/{file_name}'
    repo_url = _get_repo_url()
    path = 'tests/data/{}'.format(file_name)
    download(url_format.format(repo_url=repo_url, file_name=file_name),
             path=path,
             sha1_hash=true_data_hash[model_name])
    gt_logits = np.load(path)
    model.hybridize()
    indices = vocab[tokenizer(sentence)]
    nd_indices = mx.nd.expand_dims(mx.nd.array(indices), axis=0)
    logits, new_states = model(nd_indices, None)
    npt.assert_allclose(logits.asnumpy(), gt_logits, 1E-5, 1E-5)
