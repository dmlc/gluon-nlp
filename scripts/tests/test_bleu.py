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

"""Test BLEU."""
from __future__ import print_function

import string
import os
import re
import subprocess
import codecs
import numpy as np
from numpy.testing import assert_allclose
from ..nmt.bleu import compute_bleu, _bpe_to_words, _split_compound_word


actions = ['deletion', 'replacement', 'add']


def _sample_translation(reference, max_len):
    translation = reference[:]
    while np.random.uniform() < 0.8 and 1 < len(translation) < max_len:
        trans_len = len(translation)
        ind = np.random.randint(trans_len)
        action = np.random.choice(actions)
        if action == 'deletion':
            del translation[ind]
        elif action == 'replacement':
            ind_rep = np.random.randint(trans_len)
            translation[ind] = translation[ind_rep]
        else:
            ind_insert = np.random.randint(trans_len)
            translation.insert(ind, translation[ind_insert])
    return translation


def _sample_reference(vocabulary, k):
    return np.random.choice(vocabulary, size=k).tolist()


def _sample_translation_corpus(reference_corpus_list, max_len):
    translation_corpus = []
    for references in zip(*reference_corpus_list):
        n_refs = len(references)
        ref_ind = np.random.randint(n_refs)
        translation = _sample_translation(references[ref_ind], max_len)
        translation_corpus.append(translation)
    return translation_corpus


def _sample_reference_corpus(vocabulary, n, max_len, n_refs=5):
    reference_corpus_list = [[] for _ in range(n_refs)]
    for _ in range(n):
        for i in range(n_refs):
            ref_len = np.random.randint(1, max_len + 1)
            reference = _sample_reference(vocabulary, ref_len)
            reference_corpus_list[i].append(reference)
    return reference_corpus_list


def _write_translaton(translations, path='hypothesis'):
    out_file = codecs.open(path, 'w', 'utf-8')
    preds = [' '.join(translation) for translation in translations]
    out_file.write('\n'.join(preds) + '\n')
    out_file.flush()
    out_file.close()


def _write_reference(references, path='reference'):
    for i, reference in enumerate(references):
        out_file = codecs.open(path + str(i), 'w', 'utf-8')
        refs = [' '.join(ref) for ref in reference]
        out_file.write('\n'.join(refs) + '\n')
        out_file.flush()
        out_file.close()


def test_bleu():
    n = 100
    max_len = 50
    n_refs = 5
    path = os.path.dirname(os.path.realpath(__file__))
    ref_path = os.path.join(path, 'reference')
    trans_path = os.path.join(path, 'hypothesis')
    vocabulary = list(string.ascii_lowercase)
    reference_corpus_list = _sample_reference_corpus(vocabulary, n, max_len, n_refs)
    translation_corpus = _sample_translation_corpus(reference_corpus_list, max_len)
    _write_reference(reference_corpus_list, path=ref_path)
    _write_translaton(translation_corpus, path=trans_path)
    ret_bleu, _, _, _, _ = compute_bleu(reference_corpus_list, translation_corpus)
    mose_ret = subprocess.check_output('perl %s/multi-bleu.perl %s < %s'
                                       % (path, ref_path, trans_path),
                                       shell=True).decode('utf-8')
    m = re.search('BLEU = (.+?),', mose_ret)
    gt_bleu = float(m.group(1))
    assert_allclose(round(ret_bleu * 100, 2), gt_bleu)
    os.remove(trans_path)
    for i in range(n_refs):
        os.remove(ref_path + str(i))


def test_detok_bleu():
    path = os.path.dirname(os.path.realpath(__file__))
    ref_path = os.path.join(path, 'test_references.txt')
    trans_path = os.path.join(path, 'test_translations.txt')
    with open(trans_path) as f:
        translations = f.readlines()

    with open(ref_path) as f:
        references = f.readlines()
    ret_bleu, _, _, _, _ = compute_bleu([references], translations, tokenized=False)
    mose_ret = subprocess.check_output('perl %s/multi-bleu-detok.perl %s < %s'
                                       % (path, ref_path, trans_path),
                                       shell=True).decode('utf-8')
    m = re.search('BLEU = (.+?),', mose_ret)
    gt_bleu = float(m.group(1))
    assert_allclose(round(ret_bleu * 100, 2), gt_bleu)


def test_bpe():
    sequence = ['Th@@', 'is', 'man', 'is', 'ma@@', 'rr@@', 'ied', 'wi@@', 'th', 'her']
    gt_sequence = ['This', 'man', 'is', 'married', 'with', 'her']
    merged_sequence = _bpe_to_words(sequence)
    for gt_word, word in zip(gt_sequence, merged_sequence):
        assert gt_word == word


def test_split_compound_word():
    sequence = ['rich-text', 'man', 'feed-forward', 'yes', 'true', 'machine-learning', 'language-model']
    gt_sequence = ['rich', '##AT##-##AT##', 'text', 'man', 'feed', '##AT##-##AT##', 'forward',
                   'yes', 'true', 'machine', '##AT##-##AT##', 'learning', 'language', '##AT##-##AT##', 'model']
    split_sequence = _split_compound_word(sequence)
    for gt_word, word in zip(gt_sequence, split_sequence):
        assert gt_word == word

