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

"""Test BERTDatasetTransform."""

import numpy as np
from gluonnlp.vocab import BERTVocab
from gluonnlp.data import count_tokens, BERTTokenizer

from ..bert.data.transform import BERTDatasetTransform


def test_bert_dataset_transform():
    text_a = u'is this jacksonville ?'
    text_b = u'no it is not'
    label_cls = 0
    vocab_tokens = ['is', 'this', 'jack', '##son', '##ville', '?', 'no', 'it', 'is', 'not']

    bert_vocab = BERTVocab(count_tokens(vocab_tokens))
    tokenizer = BERTTokenizer(vocab=bert_vocab)

    # test BERTDatasetTransform for classification task
    bert_cls_dataset_t = BERTDatasetTransform(tokenizer, 15,
                                              class_labels=[label_cls], pad=True,
                                              pair=True)
    token_ids, type_ids, length, label_ids = bert_cls_dataset_t((text_a, text_b, label_cls))

    text_a_tokens = ['is', 'this', 'jack', '##son', '##ville', '?']
    text_b_tokens = ['no', 'it', 'is', 'not']
    text_a_ids = bert_vocab[text_a_tokens]
    text_b_ids = bert_vocab[text_b_tokens]

    cls_ids = bert_vocab[[bert_vocab.cls_token]]
    sep_ids = bert_vocab[[bert_vocab.sep_token]]
    pad_ids = bert_vocab[[bert_vocab.padding_token]]

    concated_ids = cls_ids + text_a_ids + sep_ids + text_b_ids + sep_ids + pad_ids
    valid_token_ids = np.array([pad_ids[0]]*15, dtype=np.int32)
    for i, x in enumerate(concated_ids):
        valid_token_ids[i] = x
    valid_type_ids = np.zeros((15,), dtype=np.int32)
    start = len(text_a_tokens) + 2
    end = len(text_a_tokens)+2+len(text_b_tokens)+1
    valid_type_ids[start:end] = 1

    assert all(token_ids == valid_token_ids)
    assert length == len(vocab_tokens) + 3
    assert all(type_ids == valid_type_ids)
    assert all(label_ids == np.array([label_cls], dtype=np.int32))

    # test BERTDatasetTransform for regression task
    label_reg = 0.2
    bert_reg_dataset_t = BERTDatasetTransform(tokenizer, 15, pad=True, pair=True)
    token_ids, type_ids, length, label_reg_val = bert_reg_dataset_t((text_a, text_b, label_reg))
    assert all(token_ids == valid_token_ids)
    assert length == len(vocab_tokens) + 3
    assert all(type_ids == valid_type_ids)
    assert all(label_reg_val == np.array([label_reg], dtype=np.float32))
