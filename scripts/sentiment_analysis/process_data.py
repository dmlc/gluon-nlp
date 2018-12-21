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

"""Load various datasets."""

import re
import time

import gluonnlp as nlp
from mxnet import nd, gluon


def _load_file(data_name):
    if data_name == 'MR':
        train_dataset = nlp.data.MR(root='data/mr')
        output_size = 2
        return train_dataset, output_size
    elif data_name == 'SST-1':
        train_dataset, test_dataset = [nlp.data.SST_1(root='data/sst-1', segment=segment)
                                       for segment in ('train', 'test')]
        output_size = 5
        return train_dataset, test_dataset, output_size
    elif data_name == 'SST-2':
        train_dataset, test_dataset = [nlp.data.SST_2(root='data/sst-2', segment=segment)
                                       for segment in ('train', 'test')]
        output_size = 2
        return train_dataset, test_dataset, output_size
    elif data_name == 'Subj':
        train_dataset = nlp.data.SUBJ(root='data/Subj')
        output_size = 2
        return train_dataset, output_size
    else:
        train_dataset, test_dataset = [nlp.data.TREC(root='data/trec', segment=segment)
                                       for segment in ('train', 'test')]
        output_size = 6
        return train_dataset, test_dataset, output_size


def _clean_str(string, data_name):
    if data_name == 'SST-1' or data_name == 'SST-2':
        string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)
        string = re.sub(r'\s{2,}', ' ', string)
        return string.strip().lower()
    else:
        string = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', string)
        string = re.sub(r'\'s', ' \'s', string)
        string = re.sub(r'\'ve', ' \'ve', string)
        string = re.sub(r'n\'t', ' n\'t', string)
        string = re.sub(r'\'re', ' \'re', string)
        string = re.sub(r'\'d', ' \'d', string)
        string = re.sub(r'\'ll', ' \'ll', string)
        string = re.sub(r',', ' , ', string)
        string = re.sub(r'!', ' ! ', string)
        string = re.sub(r'\(', ' ( ', string)
        string = re.sub(r'\)', ' ) ', string)
        string = re.sub(r'\?', ' ? ', string)
        string = re.sub(r'\s{2,}', ' ', string)
        return string.strip() if data_name == 'TREC' else string.strip().lower()


def _build_vocab(data_name, train_dataset, test_dataset):
    all_token = []
    max_len = 0
    for i, line in enumerate(train_dataset):
        train_dataset[i][0] = _clean_str(line[0], data_name)
        line = train_dataset[i][0].split()
        max_len = max_len if max_len > len(line) else len(line)
        all_token.extend(line)
    for i, line in enumerate(test_dataset):
        test_dataset[i][0] = _clean_str(line[0], data_name)
        line = test_dataset[i][0].split()
        max_len = max_len if max_len > len(line) else len(line)
        all_token.extend(line)
    vocab = nlp.Vocab(nlp.data.count_tokens(all_token))
    vocab.set_embedding(nlp.embedding.create('Word2Vec', source='GoogleNews-vectors-negative300'))
    for word in vocab.embedding._idx_to_token:
        if (vocab.embedding[word] == nd.zeros(300)).sum() == 300:
            vocab.embedding[word] = nd.random.normal(-1.0, 1.0, 300)
    vocab.embedding['<unk>'] = nd.zeros(300)
    vocab.embedding['<pad>'] = nd.zeros(300)
    vocab.embedding['<bos>'] = nd.zeros(300)
    vocab.embedding['<eos>'] = nd.zeros(300)
    print('maximum length (in tokens): ', max_len)
    return vocab, max_len


# Dataset preprocessing.
def _preprocess(x, vocab, max_len):
    data, label = x
    data = vocab[data.split()]
    data = data[:max_len] + [0] * (max_len - len(data[:max_len]))
    return data, label


def _preprocess_dataset(dataset, vocab, max_len):
    start = time.time()
    dataset = [_preprocess(d, vocab=vocab, max_len=max_len) for d in dataset]
    lengths = gluon.data.SimpleDataset([len(d[0]) for d in dataset])
    end = time.time()
    print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
    return dataset, lengths


def load_dataset(data_name):
    """Load sentiment dataset."""
    if data_name == 'MR' or data_name == 'Subj':
        train_dataset, output_size = _load_file(data_name)
        vocab, max_len = _build_vocab(data_name, train_dataset, [])
        train_dataset, train_data_lengths = _preprocess_dataset(train_dataset, vocab, max_len)
        return vocab, max_len, output_size, train_dataset, train_data_lengths
    else:
        train_dataset, test_dataset, output_size = _load_file(data_name)
        vocab, max_len = _build_vocab(data_name, train_dataset, test_dataset)
        train_dataset, train_data_lengths = _preprocess_dataset(train_dataset, vocab, max_len)
        test_dataset, test_data_lengths = _preprocess_dataset(test_dataset, vocab, max_len)
        return vocab, max_len, output_size, train_dataset, train_data_lengths, test_dataset, \
               test_data_lengths
