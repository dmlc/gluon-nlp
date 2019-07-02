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

import functools

import mxnet as mx
import pytest
from mxnet.test_utils import *

import gluonnlp as nlp


@pytest.fixture
def counter():
    return nlp.data.utils.Counter( ['a', 'b', 'b', 'c', 'c', 'c',
                                    'some_word$'])


@pytest.mark.serial
@pytest.mark.remote_required
def test_bertvocab():
    ctx = mx.cpu()

    bert_base1, vocab1 = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_cased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False,
                                             root='tests/data/model/')

    bert_base2, vocab2 = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False,
                                             root='tests/data/model/')

    bert_base3, vocab3 = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='wiki_multilingual_cased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False,
                                             root='tests/data/model/')

    bert_base4, vocab4 = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='wiki_multilingual_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False,
                                             root='tests/data/model/')

    bert_base5, vocab5 = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='wiki_cn_cased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False,
                                             root='tests/data/model/')

    assert vocab1.cls_token == vocab2.cls_token == vocab3.cls_token == \
        vocab4.cls_token == vocab5.cls_token == nlp.vocab.bert.CLS_TOKEN

    assert vocab1.sep_token == vocab2.sep_token == vocab3.sep_token == \
        vocab4.sep_token == vocab5.sep_token == nlp.vocab.bert.SEP_TOKEN

    assert vocab1.mask_token == vocab2.mask_token == vocab3.mask_token == \
        vocab4.mask_token == vocab5.mask_token == nlp.vocab.bert.MASK_TOKEN

    assert vocab1.padding_token == vocab2.padding_token == vocab3.padding_token == \
        vocab4.padding_token == vocab5.padding_token == nlp.vocab.bert.PADDING_TOKEN

    assert vocab1.unknown_token == vocab2.unknown_token == vocab3.unknown_token == \
        vocab4.unknown_token == vocab5.unknown_token == nlp.vocab.bert.UNKNOWN_TOKEN


@pytest.mark.remote_required
def test_bert_vocab_from_sentencepiece():
    # the downloaded bpe vocab includes tokens for unk and padding, but without bos/eos.
    url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-682b5d15.bpe'
    f = download(url, overwrite=True)
    bert_vocab = nlp.vocab.BERTVocab.from_sentencepiece(f, eos_token=u'<eos>')

    import sentencepiece
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(f)

    # check special tokens
    from gluonnlp.data.utils import _convert_to_unicode
    assert _convert_to_unicode(spm.IdToPiece(spm.unk_id())) == bert_vocab.unknown_token
    assert _convert_to_unicode(spm.IdToPiece(spm.pad_id())) == bert_vocab.padding_token
    assert None == bert_vocab.bos_token
    assert u'<eos>' == bert_vocab.eos_token
    assert u'<eos>' in bert_vocab
    reserved_tokens = [u'[MASK]', u'[SEP]', u'[CLS]', u'<eos>', u'[PAD]']
    assert  len(reserved_tokens) == len(bert_vocab.reserved_tokens)
    assert all(t in bert_vocab.reserved_tokens for t in reserved_tokens)
    num_tokens = len(spm)
    for i in range(num_tokens):
        token = _convert_to_unicode(spm.IdToPiece(i))
        assert bert_vocab[token] == i


@pytest.mark.parametrize('unknown_token', ['<unk>', None])
def test_bert_vocab_serialization(unknown_token):
    def check(vocab):
        assert vocab.mask_token == '[MASK]'
        assert vocab.sep_token == '[SEP]'
        assert vocab.cls_token == '[CLS]'

        if not unknown_token:
            with pytest.raises(KeyError):
                vocab['hello']
        else:
            vocab['hello']

    vocab = nlp.vocab.BERTVocab(unknown_token=unknown_token)
    check(vocab)

    loaded_vocab = nlp.vocab.BERTVocab.from_json(vocab.to_json())
    check(loaded_vocab)

    # Interoperability of serialization format with nlp.Vocab
    loaded_vocab = nlp.Vocab.from_json(vocab.to_json())
    check(loaded_vocab)
