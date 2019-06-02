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
import gluonnlp as nlp
import pytest


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

    assert nlp.vocab.BERTVocab.CLS_TOKEN == '[CLS]'
    assert nlp.vocab.BERTVocab.SEP_TOKEN == '[SEP]'
    assert nlp.vocab.BERTVocab.MASK_TOKEN == '[MASK]'
    assert nlp.vocab.BERTVocab.PADDING_TOKEN == '[PAD]'
    assert nlp.vocab.BERTVocab.UNKNOWN_TOKEN == '[UNK]'

    assert vocab1.cls_token == vocab2.cls_token == vocab3.cls_token == \
        vocab4.cls_token == vocab5.cls_token == nlp.vocab.BERTVocab.CLS_TOKEN

    assert vocab1.sep_token == vocab2.sep_token == vocab3.sep_token == \
        vocab4.sep_token == vocab5.sep_token == nlp.vocab.BERTVocab.SEP_TOKEN

    assert vocab1.mask_token == vocab2.mask_token == vocab3.mask_token == \
        vocab4.mask_token == vocab5.mask_token == nlp.vocab.BERTVocab.MASK_TOKEN

    assert vocab1.padding_token == vocab2.padding_token == vocab3.padding_token == \
        vocab4.padding_token == vocab5.padding_token == nlp.vocab.BERTVocab.PADDING_TOKEN

    assert vocab1.unknown_token == vocab2.unknown_token == vocab3.unknown_token == \
        vocab4.unknown_token == vocab5.unknown_token == nlp.vocab.BERTVocab.UNKNOWN_TOKEN
