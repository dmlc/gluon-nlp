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

import os

import mxnet as mx
import numpy as np
import numpy.testing as npt
import pytest
from mxnet.gluon.utils import _get_repo_url, download

from gluonnlp.data.transforms import GPT2BPEDetokenizer, GPT2BPETokenizer

from ..text_generation.model import get_model
from ..text_generation.model.gpt import gpt2_hparams


def verify_get_model_with_hparam_allow_override(models, hparam_allow_override, predefined_args_dict,
        mutable_args, dataset_name):

    for model in models:
        predefined_args = predefined_args_dict[model].copy()
        if hparam_allow_override:
            params_that_should_throw_exception = set()
        else:
            params_that_should_throw_exception = set(predefined_args.keys()) - set(mutable_args)
        params_that_threw_exception = set()
        for key in predefined_args:
            try:
                get_model(model, dataset_name=dataset_name,
                    hparam_allow_override=hparam_allow_override, **{key: predefined_args[key]})
            except:
                expected = not hparam_allow_override and not key in mutable_args
                params_that_threw_exception.add(key)
                assert expected

        assert params_that_threw_exception == params_that_should_throw_exception


@pytest.mark.parametrize('hparam_allow_override', [False, True])
def test_hparam_allow_override_gpt2(hparam_allow_override):
    models = ['gpt2_117m', 'gpt2_345m']
    mutable_args_of_models = ['dropout']
    predefined_args_dict = gpt2_hparams.copy()
    verify_get_model_with_hparam_allow_override(models, hparam_allow_override, predefined_args_dict,
            mutable_args_of_models, 'openai_webtext')


@pytest.mark.remote_required
@pytest.mark.parametrize('model_name', ['gpt2_117m', 'gpt2_345m'])
def test_pretrained_gpt2(model_name, tmp_path):
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
    path = os.path.join(str(tmp_path), file_name)
    download(url_format.format(repo_url=repo_url, file_name=file_name),
             path=path,
             sha1_hash=true_data_hash[model_name])
    gt_logits = np.load(path)
    model.hybridize()
    indices = vocab[tokenizer(sentence)]
    nd_indices = mx.nd.expand_dims(mx.nd.array(indices), axis=0)
    logits, new_states = model(nd_indices, None)
    npt.assert_allclose(logits.asnumpy(), gt_logits, 1E-5, 1E-5)
