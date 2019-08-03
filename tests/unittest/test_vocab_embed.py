# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import, print_function

import functools
import os
import random
import re
import sys

import numpy as np
import pytest
from mxnet import ndarray as nd
from mxnet.test_utils import *

import gluonnlp as nlp
from gluonnlp.base import _str_types


@pytest.fixture
def counter():
    return nlp.data.utils.Counter( ['a', 'b', 'b', 'c', 'c', 'c',
                                    'some_word$'])

def _get_test_str_of_tokens(token_delim, seq_delim):
    seq1 = token_delim + token_delim.join(['Life', 'is', 'great', '!']) + token_delim + seq_delim
    seq2 = token_delim + token_delim.join(['life', 'is', 'good', '.']) + token_delim + seq_delim
    seq3 = token_delim + token_delim.join(['life', "isn't", 'bad', '.']) + token_delim + seq_delim
    seqs = seq1 + seq2 + seq3
    return seqs

def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))

def _test_count_tokens(token_delim, seq_delim):
    source_str = _get_test_str_of_tokens(token_delim, seq_delim)

    tokens = list(simple_tokenize(source_str, token_delim, seq_delim))
    cnt1 = nlp.data.count_tokens(tokens, to_lower=False)
    assert cnt1 == nlp.data.utils.Counter(
        {'is': 2, 'life': 2, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1, "isn't": 1,
         'bad': 1})

    cnt2 = nlp.data.count_tokens(tokens, to_lower=True)
    assert cnt2 == nlp.data.utils.Counter(
        {'life': 3, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1, "isn't": 1, 'bad': 1}), cnt2

    counter_to_update = nlp.data.utils.Counter({'life': 2})

    cnt3 = nlp.data.utils.count_tokens(tokens, to_lower=False,
                                   counter=counter_to_update.copy())
    assert cnt3 == nlp.data.utils.Counter(
        {'is': 2, 'life': 4, '.': 2, 'Life': 1, 'great': 1, '!': 1, 'good': 1, "isn't": 1,
         'bad': 1})

    cnt4 = nlp.data.count_tokens(tokens, to_lower=True,
                             counter=counter_to_update.copy())
    assert cnt4 == nlp.data.utils.Counter(
        {'life': 5, 'is': 2, '.': 2, 'great': 1, '!': 1, 'good': 1, "isn't": 1, 'bad': 1})


def test_count_tokens():
    _test_count_tokens(' ', '\n')
    _test_count_tokens('IS', 'LIFE')


def test_vocabulary_getitem(counter):
    vocab = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unk>',
                      bos_token=None, eos_token=None, reserved_tokens=None)

    i1 = vocab['c']
    assert i1 == 2
    assert vocab.to_indices('c') == 2

    i2 = vocab[['c']]
    assert i2 == [2]
    assert vocab.to_indices(['c']) == [2]

    i3 = vocab[['<unk>', 'non-exist']]
    assert i3 == [0, 0]
    assert vocab.to_indices(['<unk>', 'non-exist']) == [0, 0]

    i4 = vocab[['a', 'non-exist', 'a', 'b']]
    assert i4 == [4, 0, 4, 3]
    assert vocab.to_indices(['a', 'non-exist', 'a', 'b']) == [4, 0, 4, 3]

    no_unk_vocab = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token=None,
                             bos_token=None, eos_token=None, reserved_tokens=None)
    assert no_unk_vocab['c'] == 1
    assert no_unk_vocab.to_indices('c') == 1

    assert no_unk_vocab[['c']] == [1]
    assert no_unk_vocab.to_indices(['c']) == [1]

    for words in [['<unk>', 'non-exist'], ['a', 'non-exist', 'a', 'b']]:
        with pytest.raises(KeyError):
            no_unk_vocab.to_indices(words)


def test_vocabulary_to_tokens(counter):
    vocab = nlp.Vocab(counter, max_size=None, min_freq=1,unknown_token='<unknown>',
                      bos_token=None, eos_token=None, reserved_tokens=None)
    i1 = vocab.to_tokens(2)
    assert i1 == 'c'

    i2 = vocab.to_tokens([2])
    assert i2 == ['c']

    i3 = vocab.to_tokens([0, 0])
    assert i3 == ['<unknown>', '<unknown>']

    i4 = vocab.to_tokens([4, 0, 4, 3])
    assert i4 == ['a', '<unknown>', 'a', 'b']

    for indices in [6, [6,7]]:
        with pytest.raises(ValueError):
            vocab.to_tokens(indices)


def test_vocabulary(counter):
    v1 = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    assert len(v1) == 5
    assert v1.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3, 'some_word$': 4}
    assert v1.idx_to_token[1] == 'c'
    assert v1.unknown_token == '<unk>'
    assert v1.reserved_tokens is None
    assert v1.embedding is None
    assert 'a' in v1
    assert v1.unknown_token in v1

    v2 = nlp.Vocab(counter, max_size=None, min_freq=2, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    assert len(v2) == 3
    assert v2.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2}
    assert v2.idx_to_token[1] == 'c'
    assert v2.unknown_token == '<unk>'
    assert v2.reserved_tokens is None
    assert v2.embedding is None
    assert 'a' not in v2
    assert v2.unknown_token in v2

    v3 = nlp.Vocab(counter, max_size=None, min_freq=100, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    assert len(v3) == 1
    assert v3.token_to_idx == {'<unk>': 0}
    assert v3.idx_to_token[0] == '<unk>'
    assert v3.unknown_token == '<unk>'
    assert v3.reserved_tokens is None
    assert v3.embedding is None
    assert 'a' not in v3

    v4 = nlp.Vocab(counter, max_size=2, min_freq=1, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    assert len(v4) == 3
    assert v4.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2}
    assert v4.idx_to_token[1] == 'c'
    assert v4.unknown_token == '<unk>'
    assert v4.reserved_tokens is None
    assert v4.embedding is None
    assert 'a' not in v4

    v5 = nlp.Vocab(counter, max_size=3, min_freq=1, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    assert len(v5) == 4
    assert v5.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3}
    assert v5.idx_to_token[1] == 'c'
    assert v5.unknown_token == '<unk>'
    assert v5.reserved_tokens is None
    assert v5.embedding is None
    assert 'a' in v5

    v6 = nlp.Vocab(counter, max_size=100, min_freq=1, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    assert len(v6) == 5
    assert v6.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3,
                               'some_word$': 4}
    assert v6.idx_to_token[1] == 'c'
    assert v6.unknown_token == '<unk>'
    assert v6.reserved_tokens is None
    assert v6.embedding is None
    assert 'a' in v6

    v7 = nlp.Vocab(counter, max_size=1, min_freq=2, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    assert len(v7) == 2
    assert v7.token_to_idx == {'<unk>': 0, 'c': 1}
    assert v7.idx_to_token[1] == 'c'
    assert v7.unknown_token == '<unk>'
    assert v7.reserved_tokens is None
    assert v7.embedding is None
    assert 'a' not in v7

    with pytest.raises(AssertionError):
        nlp.Vocab(counter, max_size=None, min_freq=0, unknown_token='<unknown>',
              reserved_tokens=['b'])
    with pytest.raises(AssertionError):
        nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unknown>',
              reserved_tokens=['b', 'b'])
    with pytest.raises(AssertionError):
        nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unknown>',
              reserved_tokens=['b', '<unknown>'])

    v8 = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unknown>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=['b'])
    assert len(v8) == 5
    assert v8.token_to_idx == {'<unknown>': 0, 'b': 1, 'c': 2, 'a': 3, 'some_word$': 4}
    assert v8.idx_to_token[1] == 'b'
    assert v8.unknown_token == '<unknown>'
    assert v8.reserved_tokens == ['b']
    assert v8.embedding is None
    assert 'a' in v8

    v9 = nlp.Vocab(counter, max_size=None, min_freq=2, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=['b', 'a'])
    assert len(v9) == 4
    assert v9.token_to_idx == {'<unk>': 0, 'b': 1, 'a': 2, 'c': 3}
    assert v9.idx_to_token[1] == 'b'
    assert v9.unknown_token == '<unk>'
    assert v9.reserved_tokens == ['b', 'a']
    assert v9.embedding is None
    assert 'a' in v9

    v10 = nlp.Vocab(counter, max_size=None, min_freq=100, unknown_token='<unk>',
                    padding_token=None, bos_token=None, eos_token=None, reserved_tokens=['b', 'c'])
    assert len(v10) == 3
    assert v10.token_to_idx == {'<unk>': 0, 'b': 1, 'c': 2}
    assert v10.idx_to_token[1] == 'b'
    assert v10.unknown_token == '<unk>'
    assert v10.reserved_tokens == ['b', 'c']
    assert v10.embedding is None
    assert 'a' not in v10

    v11 = nlp.Vocab(counter, max_size=1, min_freq=2, unknown_token='<unk>',
                    padding_token=None, bos_token=None, eos_token=None,
                    reserved_tokens=['<pad>', 'b'])
    assert len(v11) == 4
    assert v11.token_to_idx == {'<unk>': 0, '<pad>': 1, 'b': 2, 'c': 3}
    assert v11.idx_to_token[1] == '<pad>'
    assert v11.unknown_token == '<unk>'
    assert v11.reserved_tokens == ['<pad>', 'b']
    assert v11.embedding is None
    assert 'a' not in v11

    v12 = nlp.Vocab(counter, max_size=None, min_freq=2, unknown_token='b',
                    padding_token=None, bos_token=None, eos_token=None, reserved_tokens=['<pad>'])
    assert len(v12) == 3
    assert v12.token_to_idx == {'b': 0, '<pad>': 1, 'c': 2}
    assert v12.idx_to_token[1] == '<pad>'
    assert v12.unknown_token == 'b'
    assert v12.reserved_tokens == ['<pad>']
    assert v12.embedding is None
    assert 'a' not in v12

    v13 = nlp.Vocab(counter, max_size=None, min_freq=2, unknown_token='a',
                    padding_token=None, bos_token=None, eos_token=None, reserved_tokens=['<pad>'])
    assert len(v13) == 4
    assert v13.token_to_idx == {'a': 0, '<pad>': 1, 'c': 2, 'b': 3}
    assert v13.idx_to_token[1] == '<pad>'
    assert v13.unknown_token == 'a'
    assert v13.reserved_tokens == ['<pad>']
    assert v13.embedding is None
    assert 'a' in v13

    counter_tuple = nlp.data.utils.Counter([('a', 'a'), ('b', 'b'), ('b', 'b'), ('c', 'c'),
                                            ('c', 'c'), ('c', 'c'), ('some_word$', 'some_word$')])

    v14 = nlp.Vocab(counter_tuple, max_size=None, min_freq=1, unknown_token=('<unk>', '<unk>'),
                    padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    assert len(v14) == 5
    assert v14.token_to_idx == {('<unk>', '<unk>'): 0, ('c', 'c'): 1, ('b', 'b'): 2, ('a', 'a'): 3,
                                ('some_word$', 'some_word$'): 4}
    assert v14.idx_to_token[1] == ('c', 'c')
    assert v14.unknown_token == ('<unk>', '<unk>')
    assert v14.reserved_tokens is None
    assert v14.embedding is None
    assert ('a', 'a') in v14
    assert ('<unk>', '<unk>') in v14


def _mk_my_pretrain_file(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seqs = seq1 + seq2
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_pretrain_file2(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.01', '0.02', '0.03', '0.04', '0.05']) + '\n'
    seq2 = token_delim.join(['c', '0.06', '0.07', '0.08', '0.09', '0.1']) + '\n'
    seqs = seq1 + seq2
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_pretrain_file3(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seq3 = token_delim.join(['<unk1>', '1.1', '1.2', '1.3', '1.4',
                             '1.5']) + '\n'
    seqs = seq1 + seq2 + seq3
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_pretrain_file4(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.01', '0.02', '0.03', '0.04', '0.05']) + '\n'
    seq2 = token_delim.join(['c', '0.06', '0.07', '0.08', '0.09', '0.1']) + '\n'
    seq3 = token_delim.join(['<unk2>', '0.11', '0.12', '0.13', '0.14', '0.15']) + '\n'
    seqs = seq1 + seq2 + seq3
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_invalid_pretrain_file(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seq3 = token_delim.join(['c']) + '\n'
    seqs = seq1 + seq2 + seq3
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


def _mk_my_invalid_pretrain_file2(path, token_delim, pretrain_file):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)
    seq1 = token_delim.join(['a', '0.1', '0.2', '0.3', '0.4', '0.5']) + '\n'
    seq2 = token_delim.join(['b', '0.6', '0.7', '0.8', '0.9', '1.0']) + '\n'
    seq3 = token_delim.join(['c', '0.6', '0.7', '0.8']) + '\n'
    seqs = seq1 + seq2 + seq3
    with open(os.path.join(path, pretrain_file), 'w') as fout:
        fout.write(seqs)


@pytest.mark.parametrize('allow_extend', [True, False])
@pytest.mark.serial
def test_token_embedding_from_file(tmpdir, allow_extend):
    embed_root = str(tmpdir)
    embed_name = 'my_embed'
    elem_delim = '\t'
    pretrain_file = 'my_pretrain_file.txt'

    from_file = functools.partial(nlp.embedding.TokenEmbedding.from_file, allow_extend=allow_extend)

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim, pretrain_file)

    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file)

    my_embed = from_file(pretrain_file_path, elem_delim)

    assert 'a' in my_embed
    assert my_embed.unknown_token == '<unk>'
    assert my_embed.unknown_token in my_embed

    first_vec = my_embed.idx_to_vec[0]
    assert_almost_equal(first_vec.asnumpy(), np.array([0, 0, 0, 0, 0]))

    # Test properties
    assert my_embed.token_to_idx == {'<unk>': 0, 'a': 1, 'b': 2}
    assert my_embed.idx_to_token == ['<unk>', 'a', 'b']

    assert_almost_equal(my_embed.idx_to_vec.asnumpy(),
                       np.array([[0,  0,  0,  0,  0],
                                 [0.1, 0.2, 0.3, 0.4, 0.5],
                                 [0.6, 0.7, 0.8, 0.9, 1]])
                       )

    # Test __getitem__.
    unk_vec = my_embed['A']
    assert_almost_equal(unk_vec.asnumpy(), np.array([0, 0, 0, 0, 0]))

    a_vec = my_embed['a']
    assert_almost_equal(a_vec.asnumpy(), np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    my_embed = from_file(pretrain_file_path, elem_delim)
    # Test __setitem__.
    my_embed['a'] = nd.array([1, 2, 3, 4, 5])
    assert_almost_equal(my_embed['a'].asnumpy(), np.array([1, 2, 3, 4, 5]))
    if allow_extend:
        my_embed['unknown$$$'] = nd.array([0, 0, 0, 0, 0])
        assert_almost_equal(my_embed['unknown$$$'].asnumpy(), np.array([0, 0, 0, 0, 0]))
    else:
        with pytest.raises(KeyError):
            my_embed['unknown$$$'] = nd.array([0, 0, 0, 0, 0])
    with pytest.raises(AssertionError):
        my_embed['<unk>'] = nd.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    with pytest.raises(AssertionError):
        my_embed['<unk>'] = nd.array([0])

    unk_vecs = my_embed['<unk$unk@unk>', '<unk$unk@unk>']
    assert_almost_equal(unk_vecs.asnumpy(), np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))

    # Test loaded unknown vectors.
    pretrain_file2 = 'my_pretrain_file2.txt'
    _mk_my_pretrain_file3(os.path.join(embed_root, embed_name), elem_delim, pretrain_file2)
    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file2)
    my_embed2 = from_file(pretrain_file_path, elem_delim, init_unknown_vec=nd.ones, unknown_token='<unk>')
    unk_vec2 = my_embed2['<unk>']
    assert_almost_equal(unk_vec2.asnumpy(), np.array([1, 1, 1, 1, 1]))
    unk_vec2 = my_embed2['<unk$unk@unk>']
    assert_almost_equal(unk_vec2.asnumpy(), np.array([1, 1, 1, 1, 1]))

    my_embed3 = from_file(pretrain_file_path, elem_delim, init_unknown_vec=nd.ones, unknown_token='<unk1>')
    unk_vec3 = my_embed3['<unk1>']
    assert_almost_equal(unk_vec3.asnumpy(), np.array([1.1, 1.2, 1.3, 1.4, 1.5]))
    unk_vec3 = my_embed3['<unk$unk@unk>']
    assert_almost_equal(unk_vec3.asnumpy(), np.array([1.1, 1.2, 1.3, 1.4, 1.5]))

    # Test error handling.
    invalid_pretrain_file = 'invalid_pretrain_file.txt'
    _mk_my_invalid_pretrain_file(os.path.join(embed_root, embed_name), elem_delim,
                                 invalid_pretrain_file)
    pretrain_file_path = os.path.join(embed_root, embed_name, invalid_pretrain_file)
    with pytest.raises(AssertionError):
        from_file(pretrain_file_path, elem_delim)

    invalid_pretrain_file2 = 'invalid_pretrain_file2.txt'
    _mk_my_invalid_pretrain_file2(os.path.join(embed_root, embed_name), elem_delim,
                                  invalid_pretrain_file2)
    pretrain_file_path = os.path.join(embed_root, embed_name, invalid_pretrain_file2)
    with pytest.raises(AssertionError):
        from_file(pretrain_file_path, elem_delim)


def test_embedding_get_and_pretrain_file_names():
    assert len(nlp.embedding.list_sources(embedding_name='fasttext')) == 486
    assert len(nlp.embedding.list_sources(embedding_name='glove')) == 10
    assert len(nlp.embedding.list_sources(embedding_name='word2vec')) == 3

    reg = nlp.embedding.list_sources(embedding_name=None)

    assert len(reg['glove']) == 10
    assert len(reg['fasttext']) == 486
    assert len(reg['word2vec']) == 3

    with pytest.raises(KeyError):
        nlp.embedding.list_sources('unknown$$')


@pytest.mark.parametrize('allow_extend', [True, False])
def test_vocab_set_embedding_with_one_custom_embedding(tmpdir, allow_extend, counter):
    embed_root = str(tmpdir)
    embed_name = 'my_embed'
    elem_delim = '\t'
    pretrain_file = 'my_pretrain_file1.txt'

    from_file = functools.partial(nlp.embedding.TokenEmbedding.from_file, allow_extend=allow_extend)

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim, pretrain_file)

    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file)

    v1 = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=['<pad>'])
    v1_no_unk = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token=None,
                          padding_token=None, bos_token=None, eos_token=None,
                          reserved_tokens=['<pad>'])

    e1 = from_file(pretrain_file_path, elem_delim, init_unknown_vec=nd.ones)

    assert v1.embedding is None
    assert v1_no_unk.embedding is None
    v1.set_embedding(e1)
    v1_no_unk.set_embedding(e1)
    assert v1.embedding is not None
    assert v1_no_unk.embedding is not None

    # Test properties
    assert v1.embedding.token_to_idx == {'<unk>': 0, '<pad>': 1, 'c': 2, 'b': 3, 'a': 4, 'some_word$': 5}
    assert v1.embedding.idx_to_token == ['<unk>', '<pad>', 'c', 'b', 'a', 'some_word$']

    assert v1_no_unk.embedding.token_to_idx == {'<pad>': 0, 'c': 1, 'b': 2, 'a': 3, 'some_word$': 4}
    assert v1_no_unk.embedding.idx_to_token == ['<pad>', 'c', 'b', 'a', 'some_word$']

    assert_almost_equal(v1.embedding.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [0.6, 0.7, 0.8, 0.9, 1],
                                  [0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )
    assert_almost_equal(v1_no_unk.embedding.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [0.6, 0.7, 0.8, 0.9, 1],
                                  [0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )

    assert_almost_equal(v1.embedding['c'].asnumpy(),
                        np.array([1, 1, 1, 1, 1])
                        )
    assert_almost_equal(v1_no_unk.embedding['c'].asnumpy(),
                        np.array([1, 1, 1, 1, 1])
                        )

    assert_almost_equal(v1.embedding[['c']].asnumpy(),
                        np.array([[1, 1, 1, 1, 1]])
                        )
    assert_almost_equal(v1_no_unk.embedding[['c']].asnumpy(),
                        np.array([[1, 1, 1, 1, 1]])
                        )

    assert_almost_equal(v1.embedding[['a', 'not_exist']].asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [1, 1, 1, 1, 1]])
                        )
    with pytest.raises(KeyError):
        v1_no_unk.embedding['a', 'not_exist']

    assert_almost_equal(v1.embedding[['a', 'b']].asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )
    assert_almost_equal(v1_no_unk.embedding[['a', 'b']].asnumpy(),
                        np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )

    assert_almost_equal(v1.embedding[['A', 'b']].asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [0.6, 0.7, 0.8, 0.9, 1]])
                        )
    with pytest.raises(KeyError):
        v1_no_unk.embedding['A', 'b']

    v1.embedding['a'] = nd.array([2, 2, 2, 2, 2])
    v1.embedding['b'] = nd.array([3, 3, 3, 3, 3])
    v1_no_unk.embedding['a'] = nd.array([2, 2, 2, 2, 2])
    v1_no_unk.embedding['b'] = nd.array([3, 3, 3, 3, 3])

    assert_almost_equal(v1.embedding.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )

    assert_almost_equal(v1_no_unk.embedding.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )

    v1.embedding['<unk>'] = nd.array([0, 0, 0, 0, 0])
    assert_almost_equal(v1.embedding.idx_to_vec.asnumpy(),
                        np.array([[0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )
    with pytest.raises(KeyError):
        # The TokenEmbedding assigned to a vocab is never extendable
        v1_no_unk.embedding['<unk>'] = nd.array([0, 0, 0, 0, 0])
    v1.embedding['<unk>'] = nd.array([10, 10, 10, 10, 10])
    assert_almost_equal(v1.embedding.idx_to_vec.asnumpy(),
                        np.array([[10, 10, 10, 10, 10],
                                  [1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1],
                                  [3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1]])
                        )

    v1.set_embedding(None)
    assert v1.embedding is None
    v1_no_unk.set_embedding(None)
    assert v1_no_unk.embedding is None


@pytest.mark.parametrize('allow_extend', [True, False])
def test_vocab_set_embedding_with_two_custom_embeddings(tmpdir, allow_extend, counter):
    embed_root = str(tmpdir)
    embed_name = 'my_embed'
    elem_delim = '\t'
    pretrain_file1 = 'my_pretrain_file1.txt'
    pretrain_file2 = 'my_pretrain_file2.txt'

    from_file = functools.partial(nlp.embedding.TokenEmbedding.from_file, allow_extend=allow_extend)

    _mk_my_pretrain_file(os.path.join(embed_root, embed_name), elem_delim, pretrain_file1)
    _mk_my_pretrain_file2(os.path.join(embed_root, embed_name), elem_delim, pretrain_file2)

    pretrain_file_path1 = os.path.join(embed_root, embed_name, pretrain_file1)
    pretrain_file_path2 = os.path.join(embed_root, embed_name, pretrain_file2)

    my_embed1 = from_file(pretrain_file_path1, elem_delim, init_unknown_vec=nd.ones)
    my_embed2 = from_file(pretrain_file_path2, elem_delim)

    v1 = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unk>',
                   padding_token=None, bos_token=None, eos_token=None, reserved_tokens=None)
    v1.set_embedding(my_embed1, my_embed2)
    assert v1.embedding is not None
    assert v1.embedding.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3, 'some_word$': 4}
    assert v1.embedding.idx_to_token == ['<unk>', 'c', 'b', 'a', 'some_word$']

    with pytest.raises(AssertionError):
        v1.set_embedding(my_embed1, None, my_embed2)
    assert_almost_equal(v1.embedding.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0, 0, 0],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    assert_almost_equal(v1.embedding['c'].asnumpy(),
                        np.array([1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1])
                        )

    assert_almost_equal(v1.embedding[['b', 'not_exist']].asnumpy(),
                        np.array([[0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    v1.embedding['a'] = nd.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    v1.embedding['b'] = nd.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

    assert_almost_equal(v1.embedding.idx_to_vec.asnumpy(),
                        np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 0.06, 0.07, 0.08, 0.09, 0.1],
                                  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
                        )

    # Test loaded unknown tokens
    pretrain_file3 = 'my_pretrain_file3.txt'
    pretrain_file4 = 'my_pretrain_file4.txt'

    _mk_my_pretrain_file3(os.path.join(embed_root, embed_name), elem_delim, pretrain_file3)
    _mk_my_pretrain_file4(os.path.join(embed_root, embed_name), elem_delim, pretrain_file4)

    pretrain_file_path3 = os.path.join(embed_root, embed_name, pretrain_file3)
    pretrain_file_path4 = os.path.join(embed_root, embed_name, pretrain_file4)

    my_embed3 = from_file(pretrain_file_path3, elem_delim, init_unknown_vec=nd.ones,
                          unknown_token='<unk1>')
    my_embed4 = from_file(pretrain_file_path4, elem_delim, unknown_token='<unk2>')

    v2 = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unk>', padding_token=None,
                   bos_token=None, eos_token=None, reserved_tokens=None)
    v2.set_embedding(my_embed3, my_embed4)
    assert v2.embedding.token_to_idx == {'<unk>': 0, 'c': 1, 'b': 2, 'a': 3, 'some_word$': 4}
    assert v2.embedding.idx_to_token == ['<unk>', 'c', 'b', 'a', 'some_word$']
    assert_almost_equal(v2.embedding.idx_to_vec.asnumpy(),
                        np.array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15]])
                        )

    v3 = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unk1>', padding_token=None,
                   bos_token=None, eos_token=None, reserved_tokens=None)
    v3.set_embedding(my_embed3, my_embed4)
    assert v3.embedding.token_to_idx == {'<unk1>': 0, 'c': 1, 'b': 2, 'a': 3, 'some_word$': 4}
    assert v3.embedding.idx_to_token == ['<unk1>', 'c', 'b', 'a', 'some_word$']
    assert_almost_equal(v3.embedding.idx_to_vec.asnumpy(),
                        np.array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15]])
                        )

    v4 = nlp.Vocab(counter, max_size=None, min_freq=1, unknown_token='<unk2>', padding_token=None,
                   bos_token=None, eos_token=None, reserved_tokens=None)
    v4.set_embedding(my_embed3, my_embed4)
    assert v4.embedding.token_to_idx == {'<unk2>': 0, 'c': 1, 'b': 2, 'a': 3, 'some_word$': 4}
    assert v4.embedding.idx_to_token == ['<unk2>', 'c', 'b', 'a', 'some_word$']
    assert_almost_equal(v4.embedding.idx_to_vec.asnumpy(),
                        np.array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.01, 0.02, 0.03, 0.04, 0.05],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15]])
                        )

    counter2 = nlp.data.utils.Counter(['b', 'b', 'c', 'c', 'c', 'some_word$'])

    v5 = nlp.Vocab(counter2, max_size=None, min_freq=1, unknown_token='a', padding_token=None,
                   bos_token=None, eos_token=None, reserved_tokens=None)
    v5.set_embedding(my_embed3, my_embed4)
    assert v5.embedding.token_to_idx == {'a': 0, 'c': 1, 'b': 2, 'some_word$': 3}
    assert v5.embedding.idx_to_token == ['a', 'c', 'b', 'some_word$']
    assert_almost_equal(v5.embedding.idx_to_vec.asnumpy(),
                        np.array([[1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.06, 0.07, 0.08, 0.09, 0.1],
                                  [0.6, 0.7, 0.8, 0.9, 1,
                                   0.11, 0.12, 0.13, 0.14, 0.15],
                                  [1.1, 1.2, 1.3, 1.4, 1.5,
                                   0.11, 0.12, 0.13, 0.14, 0.15]])
                        )


@pytest.mark.parametrize('allow_extend', [True, False])
@pytest.mark.parametrize('unknown_token', [True, False])
@pytest.mark.parametrize('vocab_unknown_token', [True, False])
@pytest.mark.parametrize('initialize', [True, False])
def test_vocab_set_embedding_with_subword_lookup_only_token_embedding(
        allow_extend, unknown_token, vocab_unknown_token, initialize):
    embsize = 5

    class NaiveLookup:
        def __contains__(self, token):
            return True

        def __getitem__(self, tokens):
            if isinstance(tokens, _str_types):
                return nd.ones(embsize)
            else:
                return nd.ones((len(tokens), embsize))

    c = nlp.data.utils.Counter(['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])
    v = nlp.Vocab(c, max_size=None, min_freq=1,
                  unknown_token='<unk>' if vocab_unknown_token else None,
                  padding_token='<pad>')

    assert v.embedding is None

    e = nlp.embedding.TokenEmbedding(
        unknown_lookup=NaiveLookup(), allow_extend=allow_extend,
        unknown_token='<unk>' if unknown_token else None)

    if initialize and unknown_token:
        e[e.unknown_token] = nd.zeros(embsize)
    elif initialize and allow_extend:
        e["hello"] = e.unknown_lookup["hello"]
    else:  # Cannot initialize, even if initialize is True
        with pytest.raises(AssertionError):
            v.set_embedding(e)
        return  # cannot test more

    v.set_embedding(e)
    assert v.embedding is not None
    assert v.embedding.idx_to_vec is not None
    assert v.embedding.idx_to_vec.shape == (len(v), embsize)

    for t in c.keys():
        assert np.all(np.isclose(1, v.embedding[t].asnumpy()))


@pytest.mark.serial
@pytest.mark.remote_required
def test_download_embed():
    @nlp.embedding.register
    class Test(nlp.embedding.TokenEmbedding):
        # 33 bytes.
        source_file_hash = \
                {'embedding_test': ('embedding_test.vec',
                                    '29b9a6511cf4b5aae293c44a9ec1365b74f2a2f8')}
        namespace = 'test'

        def __init__(self, embedding_root='embedding', init_unknown_vec=nd.zeros, **kwargs):
            source = 'embedding_test'
            Test._check_source(self.source_file_hash, source)

            file_path = Test._get_file_path(self.source_file_hash,
                                            embedding_root, source)
            unknown_token = kwargs.pop('unknown_token', '<unk>')
            idx_to_token, idx_to_vec, unknown_token = self._load_embedding(
                file_path,
                elem_delim=' ',
                unknown_token=unknown_token,
                init_unknown_vec=init_unknown_vec)

            return super(Test, self).__init__(unknown_token=unknown_token,
                                              init_unknown_vec=None,
                                              idx_to_token=idx_to_token,
                                              idx_to_vec=idx_to_vec,
                                              **kwargs)

    test_embed = nlp.embedding.create('test', embedding_root='tests/data/embedding')
    assert_almost_equal(test_embed['hello'].asnumpy(), (nd.arange(5) + 1).asnumpy())
    assert_almost_equal(test_embed['world'].asnumpy(), (nd.arange(5) + 6).asnumpy())
    assert_almost_equal(test_embed['<unk>'].asnumpy(), nd.zeros((5,)).asnumpy())


def test_vocab_serialization():
    # Preserving unknown_token behaviour
    vocab = nlp.Vocab(unknown_token=None)
    with pytest.raises(KeyError):
        vocab['hello']
    loaded_vocab = nlp.Vocab.from_json(vocab.to_json())
    with pytest.raises(KeyError):
        loaded_vocab['hello']

    vocab = nlp.Vocab(unknown_token='abc')
    vocab['hello']
    loaded_vocab = nlp.Vocab.from_json(vocab.to_json())
    loaded_vocab['hello']


def test_token_embedding_from_serialized_file(tmpdir):
    embed_root = str(tmpdir)
    embed_name = 'my_embed'
    elem_delim = '\t'
    pretrain_file = 'my_pretrain_file.txt'
    serialize_file = 'my_pretrain_file.npz'

    _mk_my_pretrain_file(
        os.path.join(embed_root, embed_name), elem_delim, pretrain_file)

    pretrain_file_path = os.path.join(embed_root, embed_name, pretrain_file)
    serialize_file_path = os.path.join(embed_root, embed_name, serialize_file)

    # Serialize the embedding in format suitable for storage on S3 and test if
    # loading the serialized file always results in the same as loading the
    # text file would
    my_embed_for_serialization = nlp.embedding.TokenEmbedding.from_file(
        pretrain_file_path, elem_delim=elem_delim, unknown_token=None)
    my_embed_for_serialization.serialize(serialize_file_path)

    # Test w/wo unknown token
    known_unknown_token = my_embed_for_serialization.idx_to_token[-1]
    for unknown_token in [None, '<some_unknown_token>', known_unknown_token]:
        my_embed_text = nlp.embedding.TokenEmbedding.from_file(
            pretrain_file_path, elem_delim=elem_delim,
            unknown_token=unknown_token)
        my_embed_serialize = nlp.embedding.TokenEmbedding.from_file(
            serialize_file_path, unknown_token=unknown_token)
        assert my_embed_serialize == my_embed_text


@pytest.mark.parametrize('unknown_token',
                         ['<strangetoken>', None, nlp._constants.UNK_TOKEN])
@pytest.mark.serial
@pytest.mark.remote_required
def test_token_embedding_from_file_S3_with_custom_unknown_token(unknown_token):
    nlp.embedding.create('glove', source='glove.6B.50d',
                         unknown_token=unknown_token,
                         embedding_root='tests/data/embedding')


@pytest.mark.parametrize('load_ngrams', [True, False])
@pytest.mark.serial
@pytest.mark.remote_required
def test_token_embedding_from_S3_fasttext_with_ngrams(load_ngrams):
    embed = nlp.embedding.create('fasttext', source='wiki.simple',
                                 load_ngrams=load_ngrams, unknown_token=None,
                                 embedding_root='tests/data/embedding')

    if load_ngrams:
        embed['$$$unknownword$$$']
    else:
        with pytest.raises(KeyError):
            embed['$$$unknownword$$$']


@pytest.mark.parametrize('setinconstructor', [True, False])
@pytest.mark.parametrize('lookup', ['naive', 'incapable'])
@pytest.mark.parametrize('initializetokenembedding', [True, False])
@pytest.mark.parametrize('unknown_token', [True, False])
@pytest.mark.parametrize('allow_extend', [True, False])
def test_token_embedding_unknown_lookup(setinconstructor, lookup,
                                        initializetokenembedding,
                                        unknown_token, allow_extend, tmpdir):
    class NaiveLookup:
        dim = 5  # Must match _mk_my_pretrain_file

        def __contains__(self, token):
            return True

        def __getitem__(self, tokens):
            if isinstance(tokens, _str_types):
                return nd.ones(self.dim)
            else:
                return nd.ones((len(tokens), self.dim))

    class IncapableLookup:
        def __contains__(self, token):
            return False

        def __getitem__(self, tokens):
            raise KeyError

    if initializetokenembedding:
        # Load a TokenEmbedding with idx_to_vec already initialized
        embed_root = str(tmpdir)
        embed_name = 'my_embed'
        elem_delim = '\t'
        pretrain_file = 'my_pretrain_file.txt'
        _mk_my_pretrain_file(
            os.path.join(embed_root, embed_name), elem_delim, pretrain_file)
        pretrain_file_path = os.path.join(embed_root, embed_name,
                                          pretrain_file)
        TokenEmbedding = functools.partial(
            nlp.embedding.TokenEmbedding.from_file, pretrain_file_path,
            elem_delim)
    else:
        TokenEmbedding = nlp.embedding.token_embedding.TokenEmbedding

    Lookup = NaiveLookup if lookup == "naive" else IncapableLookup

    if setinconstructor:
        TokEmb = functools.partial(
            TokenEmbedding, unknown_lookup=Lookup(), allow_extend=allow_extend,
            unknown_token='<unk>' if unknown_token else None)
    else:

        def TokEmb(*args, **kwargs):
            token_embedding = TokenEmbedding(
                allow_extend=allow_extend,
                unknown_token='<unk>' if unknown_token else None, *args,
                **kwargs)
            token_embedding.unknown_lookup = Lookup()
            return token_embedding

    token_embedding = TokEmb()
    if lookup == 'incapable' and not initializetokenembedding:
        with pytest.raises(KeyError):
            token_embedding['hello']
    elif lookup == 'incapable' and initializetokenembedding and not unknown_token:
        with pytest.raises(KeyError):
            token_embedding['hello']
    elif lookup == 'incapable' and initializetokenembedding and unknown_token:
        assert 'hello' not in token_embedding.token_to_idx
        assert np.all(np.isclose(0, token_embedding['hello'].asnumpy()))
        assert 'hello' not in token_embedding.token_to_idx
    elif lookup != 'naive':
        raise RuntimeError('Invalid test parameterization.')
    else:
        assert 'hello' not in token_embedding.token_to_idx
        assert np.all(np.isclose(1, token_embedding['hello'].asnumpy()))
        assert 'hello' not in token_embedding.token_to_idx

        if allow_extend:
            token_embedding['hello'] = token_embedding.unknown_lookup['hello']
            assert 'hello' in token_embedding.token_to_idx
            assert np.all(np.isclose(1, token_embedding['hello'].asnumpy()))

            token_embedding[['hello2', 'world']] = \
                token_embedding.unknown_lookup[['hello2', 'world']]
            assert 'hello2' in token_embedding.token_to_idx
            assert 'world' in token_embedding.token_to_idx
            assert np.all(np.isclose(1, token_embedding['hello2'].asnumpy()))


@pytest.mark.parametrize('initializeidxtovecbyextending', [True, False])
def test_token_embedding_manual_extension(initializeidxtovecbyextending,
                                          tmpdir):
    if not initializeidxtovecbyextending:
        # Load a TokenEmbedding with idx_to_vec already initialized
        embed_root = str(tmpdir)
        embed_name = 'my_embed'
        elem_delim = '\t'
        pretrain_file = 'my_pretrain_file.txt'
        _mk_my_pretrain_file(
            os.path.join(embed_root, embed_name), elem_delim, pretrain_file)
        pretrain_file_path = os.path.join(embed_root, embed_name,
                                          pretrain_file)
        TokEmb = functools.partial(nlp.embedding.TokenEmbedding.from_file,
                                   pretrain_file_path, elem_delim,
                                   allow_extend=True)
    else:
        TokEmb = functools.partial(
            nlp.embedding.token_embedding.TokenEmbedding, allow_extend=True)

    # Uninitialized token_embedding._idx_to_vec based
    token_embedding = TokEmb()
    token_embedding['hello'] = nd.zeros(shape=(1, 5))
    assert np.all(np.isclose(0, token_embedding['hello'].asnumpy()))

    token_embedding = TokEmb()
    token_embedding['hello'] = nd.zeros(shape=(5, ))
    assert np.all(np.isclose(0, token_embedding['hello'].asnumpy()))

    token_embedding = TokEmb()
    token_embedding[['hello', 'world']] = nd.zeros(shape=(2, 5))
    assert np.all(np.isclose(0, token_embedding['hello'].asnumpy()))
    assert np.all(np.isclose(0, token_embedding['world'].asnumpy()))

    with pytest.raises(AssertionError):
        token_embedding = TokEmb()
        token_embedding[['hello', 'world']] = nd.zeros(shape=(1, 5))

    with pytest.raises(AssertionError):
        token_embedding = TokEmb()
        token_embedding[['hello', 'world']] = nd.zeros(shape=(5, ))

@pytest.mark.serial
@pytest.mark.remote_required
def test_token_embedding_serialization():
    @nlp.embedding.register
    class Test(nlp.embedding.TokenEmbedding):
        # 33 bytes.
        source_file_hash = \
                {'embedding_test': ('embedding_test.vec',
                                    '29b9a6511cf4b5aae293c44a9ec1365b74f2a2f8')}
        namespace = 'test'

        def __init__(self, embedding_root='tests/data/embedding', **kwargs):
            source = 'embedding_test'
            Test._check_source(self.source_file_hash, source)

            file_path = Test._get_file_path(self.source_file_hash,
                                            embedding_root, source)

            unknown_token = kwargs.pop('unknown_token', '<unk>')
            init_unknown_vec = kwargs.pop('init_unknown_vec', nd.zeros)
            idx_to_token, idx_to_vec, unknown_token = self._load_embedding(
                file_path,
                elem_delim=' ',
                unknown_token=unknown_token,
                init_unknown_vec=init_unknown_vec)

            super(Test, self).__init__(unknown_token=unknown_token,
                                       init_unknown_vec=None,
                                       idx_to_token=idx_to_token,
                                       idx_to_vec=idx_to_vec,
                                       **kwargs)


    emb = nlp.embedding.create('test', embedding_root='tests/data/embedding')

    # Test uncompressed serialization
    file_path = os.path.join('tests', 'data', 'embedding', 'embeddings.npz')
    emb.serialize(file_path, compress=False)
    loaded_emb = Test.deserialize(file_path)
    assert loaded_emb == emb

    # Test compressed serialization
    file_path_compressed = os.path.join('tests', 'data', 'embedding', 'embeddings_compressed.npz')
    emb.serialize(file_path_compressed, compress=True)
    loaded_emb = Test.deserialize(file_path)
    assert loaded_emb == emb


def test_word_embedding_evaluation_registry():
    with pytest.raises(RuntimeError):

        @nlp.embedding.evaluation.register
        class InvalidEvaluationFunction:
            pass

    with pytest.raises(KeyError):
        nlp.embedding.evaluation.create('invalid', 'InvalidEvaluationFunction')

    nlp.embedding.evaluation.list_evaluation_functions()
    nlp.embedding.evaluation.list_evaluation_functions(kind='similarity')
    nlp.embedding.evaluation.list_evaluation_functions(kind='analogy')
    with pytest.raises(KeyError):
        nlp.embedding.evaluation.list_evaluation_functions('invalid')


@pytest.mark.parametrize(
    'similarity_function',
    nlp.embedding.evaluation.list_evaluation_functions('similarity'))
@pytest.mark.serial
@pytest.mark.remote_required
def test_word_embedding_similarity_evaluation_models(similarity_function):
    try:
        from scipy import stats
    except ImportError:
        raise ImportError('This testcase requires scipy.')

    dataset = nlp.data.WordSim353()

    counter = nlp.data.utils.Counter(w for wpair in dataset for w in wpair[:2])
    vocab = nlp.vocab.Vocab(counter)
    vocab.set_embedding(
        nlp.embedding.create('fasttext', source='wiki.simple',
                             embedding_root='tests/data/embedding'))

    data = [[vocab[d[0]], vocab[d[1]], d[2]] for d in dataset]
    words1, words2, scores = zip(*data)

    evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
        vocab.embedding.idx_to_vec,
        similarity_function=similarity_function)
    evaluator.initialize()

    words1, words2 = nd.array(words1), nd.array(words2)
    pred_similarity = evaluator(words1, words2)

    sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
    assert np.isclose(0.6076485693769645, sr.correlation)


@pytest.mark.parametrize(
    'analogy_function',
    nlp.embedding.evaluation.list_evaluation_functions('analogy'))
@pytest.mark.serial
@pytest.mark.remote_required
def test_word_embedding_analogy_evaluation_models(analogy_function):
    dataset = nlp.data.GoogleAnalogyTestSet()
    dataset = [d for i, d in enumerate(dataset) if i < 10]

    embedding = nlp.embedding.create('fasttext', source='wiki.simple',
                                     embedding_root='tests/data/embedding')
    counter = nlp.data.utils.Counter(embedding.idx_to_token)
    vocab = nlp.vocab.Vocab(counter)
    vocab.set_embedding(embedding)

    dataset_coded = [[vocab[d[0]], vocab[d[1]], vocab[d[2]], vocab[d[3]]]
                     for d in dataset]
    dataset_coded_nd = nd.array(dataset_coded, dtype=np.int64)

    for k in [1, 3]:
        for exclude_question_words in [True, False]:
            evaluator = nlp.embedding.evaluation.WordEmbeddingAnalogy(
                idx_to_vec=vocab.embedding.idx_to_vec,
                analogy_function=analogy_function, k=k,
                exclude_question_words=exclude_question_words)
            evaluator.initialize()

            words1 = dataset_coded_nd[:, 0]
            words2 = dataset_coded_nd[:, 1]
            words3 = dataset_coded_nd[:, 2]
            pred_idxs = evaluator(words1, words2, words3).astype(np.int64)

            # If we don't exclude inputs most predictions should be wrong
            words4 = dataset_coded_nd[:, 3]
            accuracy = (pred_idxs[:, 0] == words4).astype(np.float64).mean()
            accuracy = accuracy.asscalar()
            if not exclude_question_words:
                assert accuracy <= 0.1

                # Instead the model would predict W3 most of the time
                accuracy_w3 = (pred_idxs[:, 0] == words3).astype(np.float64).mean()
                assert accuracy_w3.asscalar() >= 0.89

            else:
                # The wiki.simple vectors don't perform too good
                assert accuracy >= 0.29

            # Assert output shape
            assert pred_idxs.shape[1] == k


def test_subword_function_bytes():
    sf = nlp.vocab.create_subword_function('ByteSubwords')

    assert [[116, 101, 115, 116]] == sf([u'test'])
    assert [[207, 132, 206, 181, 207, 131, 207, 132]] == sf([u'τεστ'])


def test_subword_function_ngramhashes():
    num_subwords = 1000
    sf = nlp.vocab.create_subword_function('NGramHashes', ngrams=[3, 4, 5, 6],
                                           num_subwords=num_subwords)

    assert set([8, 195, 271, 500, 201, 445, 379, 831, 617, 851]) == set(sf(['test'])[0])
    assert set([8, 195, 271, 500, 201, 445, 379, 831, 617, 851]) == set(sf([u'test'])[0])
    assert set([429, 793, 101, 334, 295, 474, 145, 524, 388, 790]) == set(sf([u'τεστ'])[0])
    assert 1669484008 == sf.fasttext_hash_asbytes('<te')
    assert 1669484008 == sf.fasttext_hash_asbytes(u'<te')
    assert 2688791429 == sf.fasttext_hash_asbytes(u'<τε')
    assert 1669484008 % num_subwords == next(iter(sf.subwords_to_indices(['<te'])))
    assert 1669484008 % num_subwords == next(iter(sf.subwords_to_indices([u'<te'])))
    assert 2688791429 % num_subwords == next(iter(sf.subwords_to_indices([u'<τε'])))


@pytest.mark.parametrize('unknown_token', ['<unk>', None])
@pytest.mark.parametrize('padding_token', ['<pad>', '<eos>', None])  # padding_token == eos_token
@pytest.mark.parametrize('eos_token', ['<eos>', None])
@pytest.mark.parametrize('reserved_tokens', [['<tok>'], []])
def test_vocab_duplicate_special_tokens(unknown_token, padding_token,
                                        eos_token, reserved_tokens):
    """Different special tokens are allowed to map to the same representations.

    Special tokens are a subset of the reserved tokens. In general reserved
    tokens must not contain duplicates; however, it is allowed that multiple
    special tokens use the same reserved token.

    """
    counter = nlp.data.utils.Counter(
        ['a', 'b', 'b', 'c', 'c', 'c', 'some_word$'])

    Vocab = functools.partial(nlp.Vocab,
                              counter,
                              max_size=None,
                              min_freq=1,
                              unknown_token=unknown_token,
                              padding_token=padding_token,
                              bos_token=None,
                              eos_token=eos_token)

    v = Vocab(reserved_tokens=reserved_tokens)

    # Duplicate special tokens must not corrupt the index
    # (Broken before GluonNLP 0.7)
    if eos_token is not None and padding_token == eos_token:
        # padding_token == eos_token; there should only be a single index for
        # <eos>
        # Before GluonNLP 0.7, idx_to_token looked like
        # ['<unk>', '<eos>', '<eos>', 'c', 'b', 'a']
        # But it should look like
        # ['<unk>', '<eos>', 'c', 'b', 'a']
        assert len(v.idx_to_token) == len(v.token_to_idx)
        assert len(v.idx_to_token) == len(set(v.idx_to_token))

    # Specifying a special tokens as reserved tokens is counted as duplicate
    if eos_token is not None:
        with pytest.raises(AssertionError):
            Vocab(reserved_tokens=reserved_tokens + [eos_token])


@pytest.mark.parametrize('unknown_token', ['<unk>', None])
@pytest.mark.parametrize('padding_token', ['<pad>', None])
def test_vocab_identifiers_to_tokens_sanity_checks(unknown_token,
                                                   padding_token, counter):
    Vocab = functools.partial(nlp.Vocab,
                              counter,
                              max_size=None,
                              min_freq=1,
                              unknown_token=unknown_token,
                              bos_token=None,
                              eos_token=None)
    # Special tokens are automatically added
    v = Vocab(my_token='<does_not_exist_yet>')
    assert v.my_token == '<does_not_exist_yet>'

    # Special token names must end in _token
    with pytest.raises(ValueError):
        Vocab(special_tok='<token>')

    # Cannot set internals
    with pytest.raises(ValueError):
        Vocab(_private_token='<token>')

    # Enforces uniqueness requirement of reserved_tokens argument
    with pytest.raises(AssertionError):
        Vocab(reserved_tokens=['<token>'], special_token='<token>')

    # Many-to-one mapping is allowed
    v = Vocab(first_name_of_token='<token>', second_name_of_token='<token>')
    assert v.first_name_of_token == '<token>'
    assert v.second_name_of_token == '<token>'
    if unknown_token:
        v = Vocab(unk_token=unknown_token)
        assert v.unk_token == unknown_token
        assert v.unk_token == v.unknown_token
    if padding_token:
        v = Vocab(pad_token=padding_token)
        assert v.pad_token == padding_token
        assert v.pad_token == v.padding_token


@pytest.mark.parametrize('unknown_token', ['<unk>', None])
@pytest.mark.parametrize('padding_token', ['<pad>', None])
@pytest.mark.parametrize('identifiers_to_tokens', [{
    'important_token': '<imp>'
}, {}])
@pytest.mark.parametrize('test_serialization', [True, False])
def test_vocab_identifiers_to_tokens(unknown_token, padding_token,
                                     identifiers_to_tokens, test_serialization,
                                     counter):
    vocab = nlp.Vocab(counter,
                      max_size=None,
                      min_freq=1,
                      unknown_token=unknown_token,
                      padding_token=padding_token,
                      bos_token=None,
                      eos_token=None,
                      **identifiers_to_tokens)

    if test_serialization:
        vocab = nlp.Vocab.from_json(vocab.to_json())

    if identifiers_to_tokens:
        for identifier, token in identifiers_to_tokens.items():
            assert hasattr(vocab, identifier)
            assert getattr(vocab, identifier) == token
            assert token in vocab.reserved_tokens

    assert getattr(vocab, 'unknown_token') == unknown_token
    assert getattr(vocab, 'padding_token') == padding_token


@pytest.mark.parametrize('unknown_token', ['<unk>', None])
@pytest.mark.parametrize('padding_token', ['<pad>', None])
def test_vocab_token_to_idx(unknown_token, padding_token, counter):
    reserved_tokens = ['<tok>']
    Vocab = functools.partial(nlp.Vocab,
                              counter,
                              max_size=None,
                              min_freq=1,
                              unknown_token=unknown_token,
                              padding_token=padding_token,
                              bos_token=None,
                              eos_token=None,
                              reserved_tokens=reserved_tokens)
    tokens = set(counter)
    if unknown_token is not None:
        tokens.add(unknown_token)
    if padding_token is not None:
        tokens.add(padding_token)
    if isinstance(reserved_tokens, dict):
        tokens.update(reserved_tokens.values())
    elif isinstance(reserved_tokens, list):
        tokens.update(reserved_tokens)

    # Test sanity-checks
    valid_token = next(counter.elements())
    invalid_token = 'token_that_does_not_occur_in_vocab'
    assert invalid_token not in counter
    with pytest.raises(ValueError):
        Vocab(token_to_idx={invalid_token: 0})
    with pytest.raises(ValueError):
        Vocab(token_to_idx={valid_token: -1})
    with pytest.raises(ValueError):
        Vocab(token_to_idx={valid_token: len(tokens)})

    def token_idx_check(token, idx):
        assert v[token] == idx
        assert v.token_to_idx[token] == idx
        assert v.idx_to_token[idx] == token

    def consistency_check(v):
        assert set(v.idx_to_token) == set(v.token_to_idx.keys())
        assert set(v.token_to_idx.keys()) == set(tokens)
        assert set(v.token_to_idx.values()) == set(range(len(tokens)))

    # Manual checks with special tokens
    if unknown_token:
        v = Vocab(token_to_idx={unknown_token: len(tokens) - 1})
        consistency_check(v)
        token_idx_check(unknown_token, len(tokens) -1)
    if padding_token:
        v = Vocab(token_to_idx={padding_token: len(tokens) - 1})
        consistency_check(v)
        token_idx_check(padding_token, len(tokens) -1)

    # Test 10 random user-specified indices for a subset of tokens
    for i in range(10):
        k = random.randint(0, len(tokens) - 1)
        token_to_idx = {
            k: v
            for k, v in zip(random.sample(tokens, k), random.sample(
                range(k), k))
        }
        v = Vocab(token_to_idx=token_to_idx)
        consistency_check(v)
        for token, idx in token_to_idx.items():
            token_idx_check(token, idx)


@pytest.mark.parametrize('unknown_token', ['<unk>', None])
@pytest.mark.parametrize('padding_token', ['<pad>', '<eos>', None])
@pytest.mark.parametrize('eos_token', ['<eos>', None])
@pytest.mark.parametrize('reserved_tokens', [['<tok>'], []])
def test_vocab_duplicate_special_tokens(unknown_token, padding_token,
                                        eos_token, reserved_tokens, counter):
    """Different special tokens are allowed to map to the same representations.

    Special tokens are a subset of the reserved tokens. In general reserved
    tokens must not contain duplicates; however, it is allowed that multiple
    special tokens use the same reserved token.

    """
    Vocab = functools.partial(nlp.Vocab,counter,
                  max_size=None,
                  min_freq=1,
                  unknown_token=unknown_token,
                  padding_token=padding_token,
                  bos_token=None,
                  eos_token=eos_token
                  )

    v = Vocab(reserved_tokens=reserved_tokens)

    # Specifying a special tokens as reserved tokens is counted as duplicate
    if eos_token is not None:
        with pytest.raises(AssertionError):
            Vocab(reserved_tokens=reserved_tokens + [eos_token])


def test_vocab_backwards_compatibility_prior_v0_7_corrupted_index_bug():
    with open('tests/data/vocab/backward_compat_0_7_corrupted_index', 'r') as f:
        v = nlp.Vocab.from_json(f.read())

    assert len(set(v.idx_to_token)) == len(v.token_to_idx)
    assert v['<unk>'] == 0
    assert v['<bos>'] == 2
    assert v['<eos>'] == 3
    assert v['token'] == 4

    assert v.idx_to_token[0] == '<unk>'
    assert v.idx_to_token[1] == '<eos>'  # corruption preserved for backward
                                         # compatibility
    assert v.idx_to_token[2] == '<bos>'
    assert v.idx_to_token[3] == '<eos>'
    assert v.idx_to_token[4] == 'token'


@pytest.mark.parametrize('unknown_token', ['<unk>', '<UNK>'])
@pytest.mark.parametrize('padding_token', ['<pad>', '<eos>', None])
@pytest.mark.parametrize('eos_token', ['<eos>', None])
@pytest.mark.parametrize('reserved_tokens', [['<tok>'], []])
def test_vocab_remapped_unknown_token_idx(unknown_token, padding_token, eos_token, reserved_tokens,
                                          counter):
    Vocab = functools.partial(nlp.Vocab, counter, max_size=None, min_freq=1,
                              unknown_token=unknown_token, padding_token=padding_token,
                              bos_token=None, eos_token=eos_token)

    v = Vocab()
    assert v['UNKNOWNWORD'] == 0

    v = Vocab(token_to_idx={unknown_token: 1})
    assert v['UNKNOWNWORD'] == 1
