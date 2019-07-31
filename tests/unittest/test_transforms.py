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

import mxnet as mx
import numpy as np
import pytest
from mxnet.gluon.data import SimpleDataset
from mxnet.gluon.utils import download
from numpy.testing import assert_allclose

from gluonnlp.data import count_tokens
from gluonnlp.data import transforms as t
from gluonnlp.model.utils import _load_vocab
from gluonnlp.vocab import BERTVocab, Vocab


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


@pytest.mark.skipif(sys.version_info < (3, 0),
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


@pytest.mark.skipif(sys.version_info < (3, 0),
                    reason="requires python3 or higher")
def test_moses_detokenizer():
    detokenizer = t.SacreMosesDetokenizer(return_str=False)
    text = ['Introducing', 'Gluon', ':', 'An', 'Easy-to-Use', 'Programming',
            'Interface', 'for', 'Flexible', 'Deep', 'Learning', '.']
    try:
        ret = detokenizer(text)
    except ImportError:
        warnings.warn("NLTK not installed, skip test_moses_detokenizer().")
        return
    assert isinstance(ret, list)
    assert len(ret) > 0


@pytest.mark.remote_required
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
    assert all(t in tokenizer.tokens for t in ret)
    assert len(ret) > 0
    assert text == detext


@pytest.mark.remote_required
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
    assert all(t in tokenizer.tokens for ret in reg_ret for t in ret)
    assert all(detokenizer(reg_ret[i]) == detext for i in range(len(reg_ret)))


def test_bertbasictokenizer():
    tokenizer = t.BERTBasicTokenizer(lower=True)

    # test lower_case=True
    assert tokenizer(u" \tHeLLo!how  \n Are yoU?  ") == [
        "hello", "!", "how", "are", "you", "?"]
    assert tokenizer(u"H\u00E9llo") == ["hello"]

    # test chinese
    assert tokenizer(u"ah\u535A\u63A8zz") == [
        u"ah", u"\u535A", u"\u63A8", u"zz"]

    # test is_whitespace
    assert tokenizer._is_whitespace(u" ") == True
    assert tokenizer._is_whitespace(u"\t") == True
    assert tokenizer._is_whitespace(u"\r") == True
    assert tokenizer._is_whitespace(u"\n") == True
    assert tokenizer._is_whitespace(u"\u00A0") == True
    assert tokenizer._is_whitespace(u"A") == False
    assert tokenizer._is_whitespace(u"-") == False

    # test is_control
    assert tokenizer._is_control(u"\u0005") == True
    assert tokenizer._is_control(u"A") == False
    assert tokenizer._is_control(u" ") == False
    assert tokenizer._is_control(u"\t") == False
    assert tokenizer._is_control(u"\r") == False

    # test is_punctuation
    assert tokenizer._is_punctuation(u"-") == True
    assert tokenizer._is_punctuation(u"$") == True
    assert tokenizer._is_punctuation(u"`") == True
    assert tokenizer._is_punctuation(u".") == True
    assert tokenizer._is_punctuation(u"A") == False
    assert tokenizer._is_punctuation(u" ") == False

    # test lower_case=False
    tokenizer = t.BERTBasicTokenizer(lower=False)
    assert tokenizer(u" \tHeLLo!how  \n Are yoU?  ") == [
        "HeLLo", "!", "how", "Are", "yoU", "?"]


def test_berttokenizer():

    # test WordpieceTokenizer
    vocab_tokens = ["want", "##want", "##ed", "wa", "un", "runn", "##ing"]
    vocab = Vocab(
        count_tokens(vocab_tokens),
        reserved_tokens=["[CLS]", "[SEP]"],
        unknown_token="[UNK]", padding_token=None, bos_token=None, eos_token=None)
    tokenizer = t.BERTTokenizer(vocab=vocab)

    assert tokenizer(u"unwanted running") == [
        "un", "##want", "##ed", "runn", "##ing"]
    assert tokenizer(u"unwantedX running") == ["[UNK]", "runn", "##ing"]
    assert tokenizer.is_first_subword('un')
    assert not tokenizer.is_first_subword('##want')

    # test BERTTokenizer
    vocab_tokens = ["[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
                    "##ing", ","]

    vocab = Vocab(
        count_tokens(vocab_tokens),
        reserved_tokens=["[CLS]", "[SEP]"],
        unknown_token="[UNK]", padding_token=None, bos_token=None, eos_token=None)
    tokenizer = t.BERTTokenizer(vocab=vocab)
    tokens = tokenizer(u"UNwant\u00E9d,running")
    assert tokens == ["un", "##want", "##ed", ",", "runn", "##ing"]


def test_bert_sentences_transform():
    text_a = u'is this jacksonville ?'
    text_b = u'no it is not'
    vocab_tokens = ['is', 'this', 'jack', '##son', '##ville', '?', 'no', 'it', 'is', 'not']

    bert_vocab = BERTVocab(count_tokens(vocab_tokens))
    tokenizer = t.BERTTokenizer(vocab=bert_vocab)

    # test BERTSentenceTransform
    bert_st = t.BERTSentenceTransform(tokenizer, 15, pad=True, pair=True)
    token_ids, length, type_ids = bert_st((text_a, text_b))

    text_a_tokens = ['is', 'this', 'jack', '##son', '##ville', '?']
    text_b_tokens = ['no', 'it', 'is', 'not']
    text_a_ids = bert_vocab[text_a_tokens]
    text_b_ids = bert_vocab[text_b_tokens]

    cls_ids = bert_vocab[[bert_vocab.cls_token]]
    sep_ids = bert_vocab[[bert_vocab.sep_token]]
    pad_ids = bert_vocab[[bert_vocab.padding_token]]

    concated_ids = cls_ids + text_a_ids + sep_ids + text_b_ids + sep_ids + pad_ids
    valid_token_ids = np.array([pad_ids[0]] * 15, dtype=np.int32)
    for i, x in enumerate(concated_ids):
        valid_token_ids[i] = x
    valid_type_ids = np.zeros((15,), dtype=np.int32)
    start = len(text_a_tokens) + 2
    end = len(text_a_tokens) + 2 + len(text_b_tokens) + 1
    valid_type_ids[start:end] = 1

    assert all(token_ids == valid_token_ids)
    assert length == len(vocab_tokens) + 3
    assert all(type_ids == valid_type_ids)


@pytest.mark.remote_required
def test_bert_sentencepiece_sentences_transform():
    url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-682b5d15.bpe'
    f = download(url, overwrite=True)
    bert_vocab = BERTVocab.from_sentencepiece(f)
    bert_tokenizer = t.BERTSPTokenizer(f, bert_vocab, lower=True)
    assert bert_tokenizer.is_first_subword(u'▁this')
    assert not bert_tokenizer.is_first_subword(u'this')
    max_len = 36
    data_train_raw = SimpleDataset(
        [[u'This is a very awesome, life-changing sentence.']])
    transform = t.BERTSentenceTransform(bert_tokenizer,
                                        max_len,
                                        pad=True,
                                        pair=False)
    try:
        data_train = data_train_raw.transform(transform)
    except ImportError:
        warnings.warn(
            "Sentencepiece not installed, skip test_bert_sentencepiece_sentences_transform()."
        )
        return
    processed = list(data_train)[0]

    tokens = [
        u'▁this', u'▁is', u'▁a', u'▁very', u'▁a', u'w', u'es', u'om', u'e', u'▁', u',',
        u'▁life', u'▁', u'-', u'▁c', u'hang', u'ing', u'▁sentence', u'▁', u'.'
    ]
    token_ids = [bert_vocab[bert_vocab.cls_token]
                 ] + bert_tokenizer.convert_tokens_to_ids(tokens) + [
                     bert_vocab[bert_vocab.sep_token]
                 ]
    token_ids += [bert_vocab[bert_vocab.padding_token]
                  ] * (max_len - len(token_ids))

    # token ids
    assert all(processed[0] == np.array(token_ids, dtype='int32'))
    # sequence length
    assert np.asscalar(processed[1]) == len(tokens) + 2
    # segment id
    assert all(processed[2] == np.array([0] * max_len, dtype='int32'))


@pytest.mark.remote_required
def test_gpt2_transforms():
    tokenizer = t.GPT2BPETokenizer()
    detokenizer = t.GPT2BPEDetokenizer()
    vocab = _load_vocab('openai_webtext', None, root=os.path.join('tests', 'data', 'models'))
    s = ' natural language processing tools such as gluonnlp and torchtext'
    subwords = tokenizer(s)
    indices = vocab[subwords]
    gt_gpt2_subword = [u'Ġnatural', u'Ġlanguage', u'Ġprocessing', u'Ġtools',
                       u'Ġsuch', u'Ġas', u'Ġgl', u'u', u'on',
                       u'nl', u'p', u'Ġand', u'Ġtorch', u'text']
    gt_gpt2_idx = [3288, 3303, 7587, 4899, 884, 355, 1278, 84, 261, 21283, 79, 290, 28034, 5239]
    for lhs, rhs in zip(indices, gt_gpt2_idx):
        assert lhs == rhs
    for lhs, rhs in zip(subwords, gt_gpt2_subword):
        assert lhs == rhs

    recovered_sentence = detokenizer([vocab.idx_to_token[i] for i in indices])
    assert recovered_sentence == s
