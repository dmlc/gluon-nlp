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

import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
import mxnet as mx
from mxnet.gluon.utils import download
from gluonnlp.data import transforms as t


def test_clip_sequence():
    for length in [10, 200]:
        clip_seq = t.ClipSequence(length=length)
        for seq_length in [1, 20, 500]:
            dat_npy = np.random.uniform(0, 1, (seq_length,))
            ret1 = clip_seq(dat_npy.tolist())
            assert(len(ret1) == min(seq_length, length))
            assert_allclose(np.array(ret1), dat_npy[:length])
            ret2 = clip_seq(mx.nd.array(dat_npy)).asnumpy()
            assert_allclose(ret2, dat_npy[:length])
            ret3 = clip_seq(dat_npy)
            assert_allclose(ret3, dat_npy[:length])


def test_pad_sequence():
    def np_gt(data, length, clip, pad_val):
        if data.shape[0] >= length:
            if clip:
                return data[:length]
            else:
                return data
        else:
            pad_width = [(0, length - data.shape[0])] + [(0, 0) for _ in range(data.ndim - 1)]
            return np.pad(data, mode='constant', pad_width=pad_width, constant_values=pad_val)

    for clip in [False, True]:
        for length in [5, 20]:
            for pad_val in [-1.0, 0.0, 1.0]:
                pad_seq = t.PadSequence(length=length, clip=clip, pad_val=pad_val)
                for seq_length in range(1, 100, 10):
                    for additional_shape in [(), (5,), (4, 3)]:
                        dat_npy = np.random.uniform(0, 1, (seq_length,) + additional_shape)
                        gt_npy = np_gt(dat_npy, length, clip, pad_val)
                        ret_npy = pad_seq(dat_npy)
                        ret_mx = pad_seq(mx.nd.array(dat_npy)).asnumpy()
                        assert_allclose(ret_npy, gt_npy)
                        assert_allclose(ret_mx, gt_npy)
                        if len(additional_shape) == 0:
                            ret_l = np.array(pad_seq(dat_npy.tolist()))
                            assert_allclose(ret_l, gt_npy)


@pytest.mark.skipif(sys.version_info < (3,0),
                    reason="requires python3 or higher")
def test_moses_tokenizer():
    tokenizer = t.SacreMosesTokenizer()
    text = u"Introducing Gluon: An Easy-to-Use Programming Interface for Flexible Deep Learning."
    try:
        ret = tokenizer(text)
    except ImportError:
        warnings.warn("NLTK not installed, skip test_moses_tokenizer().")
        return
    assert isinstance(ret, list)
    assert len(ret) > 0


def test_spacy_tokenizer():
    tokenizer = t.SpacyTokenizer()
    text = u"Introducing Gluon: An Easy-to-Use Programming Interface for Flexible Deep Learning."
    try:
        ret = tokenizer(text)
    except ImportError:
        warnings.warn("Spacy not installed, skip test_spacy_tokenizer().")
        return
    assert isinstance(ret, list)
    assert len(ret) > 0


@pytest.mark.skipif(sys.version_info < (3,0),
                    reason="requires python3 or higher")
def test_moses_detokenizer():
    detokenizer = t.SacreMosesDetokenizer()
    text = ['Introducing', 'Gluon', ':', 'An', 'Easy-to-Use', 'Programming',
            'Interface', 'for', 'Flexible', 'Deep', 'Learning', '.']
    try:
        ret = detokenizer(text)
    except ImportError:
        warnings.warn("NLTK not installed, skip test_moses_detokenizer().")
        return
    assert isinstance(ret, list)
    assert len(ret) > 0

def test_sentencepiece_tokenizer():
    url_format = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/vocab/{}'
    filename = 'test-0690baed.bpe'
    download(url_format.format(filename), path=os.path.join('tests', 'data', filename))
    tokenizer = t.SentencepieceTokenizer(os.path.join('tests', 'data', filename))
    detokenizer = t.SentencepieceDetokenizer(os.path.join('tests', 'data', filename))
    text = "Introducing Gluon: An Easy-to-Use Programming Interface for Flexible Deep Learning."
    try:
        ret = tokenizer(text)
        detext = detokenizer(ret)
    except ImportError:
        warnings.warn("Sentencepiece not installed, skip test_sentencepiece_tokenizer().")
        return
    assert isinstance(ret, list)
    assert len(ret) > 0
    assert text == detext

def test_sentencepiece_tokenizer_subword_regularization():
    url_format = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/vocab/{}'
    filename = 'test-31c8ed7b.uni'
    download(url_format.format(filename), path=os.path.join('tests', 'data', filename))
    tokenizer = t.SentencepieceTokenizer(os.path.join('tests', 'data', filename),
                                         -1, 0.1)
    detokenizer = t.SentencepieceDetokenizer(os.path.join('tests', 'data', filename))
    text = "Introducing Gluon: An Easy-to-Use Programming Interface for Flexible Deep Learning."
    try:
        reg_ret = [tokenizer(text) for _ in range(10)]
        detext = detokenizer(reg_ret[0])
    except ImportError:
        warnings.warn("Sentencepiece not installed, skip test_sentencepiece_tokenizer().")
        return
    assert text == detext
    assert any(reg_ret[i] != reg_ret[0] for i in range(len(reg_ret)))
    assert all(detokenizer(reg_ret[i]) == detext for i in range(len(reg_ret)))
