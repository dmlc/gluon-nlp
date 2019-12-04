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

"""Test BERTStyleDataSetTransform."""

import numpy as np
from gluonnlp.vocab import BERTVocab
from gluonnlp.data import count_tokens, BERTTokenizer, \
    BertStyleGlueTransform, BertStyleSQuADTransform, TruncateTransform, \
        ConcatSeqTransform


def test_bertstyle_glue_dataset_transform():
    text_a = u'is this jacksonville ?'
    text_b = u'no it is not'
    text_ab = u'is this jacksonville ? no it is not'
    label_cls = 0
    vocab_tokens = ['is', 'this', 'jack', '##son', '##ville', '?', 'no', 'it', 'is', 'not']

    bert_vocab = BERTVocab(count_tokens(vocab_tokens))
    tokenizer = BERTTokenizer(vocab=bert_vocab)

    # test Transform for classification task
    bert_cls_dataset_t = BertStyleGlueTransform(tokenizer, 15, class_labels=[label_cls])

    token_ids, type_ids, label_ids = bert_cls_dataset_t((text_a, text_b, label_cls))
    text_a_tokens = ['is', 'this', 'jack', '##son', '##ville', '?']
    text_b_tokens = ['no', 'it', 'is', 'not']
    text_a_ids = bert_vocab[text_a_tokens]
    text_b_ids = bert_vocab[text_b_tokens]

    cls_ids = bert_vocab[[bert_vocab.cls_token]]
    sep_ids = bert_vocab[[bert_vocab.sep_token]]

    concated_ids = cls_ids + text_a_ids + sep_ids + text_b_ids + sep_ids

    valid_type_ids = np.zeros((13,), dtype=np.int32)
    start = len(text_a_tokens) + 2
    end = len(text_a_tokens)+2+len(text_b_tokens)+1
    valid_type_ids[start:end] = 1
    assert all(token_ids == concated_ids)
    assert all(type_ids == valid_type_ids)
    assert all(label_ids == np.array([label_cls], dtype=np.int32))

    #test Transform for regression task
    label_reg = 0.2
    bert_reg_dataset_t = BertStyleGlueTransform(tokenizer, 15)
    token_ids, type_ids, label_reg_val = bert_reg_dataset_t((text_a, text_b, label_reg))
    assert all(token_ids == concated_ids)
    assert all(type_ids == valid_type_ids)
    assert all(label_reg_val == np.array([label_reg], dtype=np.float32))

    #test Transform for single input sequence
    label_reg = 0.2
    bert_reg_dataset_t = BertStyleGlueTransform(tokenizer, 15)
    token_ids, type_ids, label_reg_val = bert_reg_dataset_t((text_ab, label_reg))
    concated_ids = cls_ids + text_a_ids + text_b_ids + sep_ids

    valid_type_ids = np.zeros((12,), dtype=np.int32)
    assert all(token_ids == np.array(concated_ids))
    assert all(type_ids == valid_type_ids)
    assert all(label_reg_val == np.array([label_reg], dtype=np.float32))

def test_bertstyle_squad_dataset_transform():
    data_without_impossible = (0,
            '1',
            'what is my name?',
            'my name is jack',
            ['jack'],
            [11],
            False)

    data_with_impossible = (0,
                            '1',
                            'what is my name?',
                            'my name is jack',
                            ['John'],
                            [0],
                            True)

    vocab_tokens = ['what', 'is', 'my', 'na', '##me', '?', 'my', 'na', '##me', 'is', 'jack']
    bert_vocab = BERTVocab(count_tokens(vocab_tokens))
    tokenizer = BERTTokenizer(vocab=bert_vocab)
    trans = BertStyleSQuADTransform(tokenizer, max_seq_length=len(vocab_tokens) + 3,
                             doc_stride=3, max_query_length=6,
                             is_training=True)
    example_id, inputs, token_types, p_mask, start_label, end_label, is_impossible = \
    trans(data_without_impossible)[0]
    text_a_tokens = ['what', 'is', 'my', 'na','##me', '?']
    text_b_tokens = ['my', 'na', '##me', 'is', 'jack']
    text_a_ids = bert_vocab[text_a_tokens]
    text_b_ids = bert_vocab[text_b_tokens]

    cls_ids = bert_vocab[[bert_vocab.cls_token]]
    sep_ids = bert_vocab[[bert_vocab.sep_token]]
    concated_ids = cls_ids + text_a_ids + sep_ids + text_b_ids + sep_ids
    inputs = np.array(inputs)
    concated_ids = np.array(concated_ids)
    valid_token_type =np.ones((len(vocab_tokens) + 3,), dtype=np.int32)
    start, end = 0, len(text_a_tokens) + 2
    valid_token_type[start:end] = 0

    p_mask_valid = np.zeros((len(vocab_tokens) + 3,), dtype=np.int32)
    p_mask_valid[len(text_a_tokens) + 1] = 1
    p_mask_valid[-1] = 1

    assert all(inputs == concated_ids)
    assert example_id == data_with_impossible[0]
    assert all(token_types == valid_token_type)
    assert all(p_mask == p_mask_valid)
    assert start_label == 12
    assert end_label == 12
    assert is_impossible == False

    #squad2 with impossible
    example_id, inputs, token_types, p_mask, start_label, end_label, is_impossible = \
        trans(data_with_impossible)[0]
    assert all(inputs == concated_ids)
    assert example_id == data_with_impossible[0]
    assert all(token_types == valid_token_type)
    assert all(p_mask == p_mask_valid)
    assert start_label == 0
    assert end_label == 0
    assert is_impossible == True

def test_truncate():
    seqs = [[j*i for j in range(i)] for i in range(1,10)]
    res1 = [[0], [0, 2], [0, 3, 6], [0, 4, 8], [0, 5, 10], [0, 6], [0, 7], [0, 8], [0, 9]]
    seq = [i for i in range(20)]
    res2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    trunc = TruncateTransform(20)
    assert all(np.array(trunc(seqs)) == np.array(res1))
    assert all(np.array(trunc(seq[0])) == np.array(res2))

def test_concat_sequence():
    seqs = [[3 * i + j for j in range(3)] for i in range(3)]
    start_token = -1
    end_token = 999
    middle_tokens = ['a', 'b', 'c']
    concat = ConcatSeqTransform(start_token=start_token, token_after_seg=middle_tokens, end_token=end_token)
    res = concat(seqs)
    assert res[0] == [-1, 0, 1, 2, 'a', 3, 4, 5, 'b', 6, 7, 8, 'c', 999]
    assert res[1] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert res[2] == 14
    assert res[3] == [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]

    middle_tokens = ['a', None, 'b']
    concat = ConcatSeqTransform(start_token=start_token, token_after_seg=middle_tokens, end_token=end_token)
    res = concat(seqs)
    assert res[0] == [-1, 0, 1, 2, 'a', 3, 4, 5, 6, 7, 8, 'b', 999]
    assert res[1] == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    assert res[2] == 13
    assert res[3] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]

