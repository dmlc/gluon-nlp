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

# pylint: disable=
"""Machine translation datasets."""

__all__ = ['IWSLT2015', 'WMT2016BPE']


import os
import zipfile
import shutil
import io

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
from mxnet.gluon.data import ArrayDataset

from .dataset import TextLineDataset
from ..vocab import Vocab
from .registry import register


def _get_pair_key(src_lang, tgt_lang):
    return '_'.join(sorted([src_lang, tgt_lang]))


class _TranslationDataset(ArrayDataset):
    def __init__(self, namespace, segment, src_lang, tgt_lang, root):
        assert _get_pair_key(src_lang, tgt_lang) in self._archive_file, \
            'The given language combination: src_lang={}, tgt_lang={}, is not supported. ' \
            'Only supports language pairs = {}.'.format(
                src_lang, tgt_lang, str(self._archive_file.keys()))
        if isinstance(segment, str):
            assert segment in self._supported_segments, \
                'Only supports {} for the segment. Received segment={}'.format(
                    self._supported_segments, segment)
        else:
            for ele_segment in segment:
                assert ele_segment in self._supported_segments, \
                    'segment should only contain elements in {}. Received segment={}'.format(
                        self._supported_segments, segment)
        self._namespace = 'gluon/dataset/{}'.format(namespace)
        self._segment = segment
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._src_vocab = None
        self._tgt_vocab = None
        self._pair_key = _get_pair_key(src_lang, tgt_lang)
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        if isinstance(segment, str):
            segment = [segment]
        src_corpus = []
        tgt_corpus = []
        for ele_segment in segment:
            [src_corpus_path, tgt_corpus_path] = self._get_data(ele_segment)
            src_corpus.extend(TextLineDataset(src_corpus_path))
            tgt_corpus.extend(TextLineDataset(tgt_corpus_path))
        # Filter 0-length src/tgt sentences
        src_lines = []
        tgt_lines = []
        for src_line, tgt_line in zip(list(src_corpus), list(tgt_corpus)):
            if len(src_line) > 0 and len(tgt_line) > 0:
                src_lines.append(src_line)
                tgt_lines.append(tgt_line)
        super(_TranslationDataset, self).__init__(src_lines, tgt_lines)

    def _fetch_data_path(self, file_name_hashs):
        archive_file_name, archive_hash = self._archive_file[self._pair_key]
        paths = []
        root = self._root
        for data_file_name, data_hash in file_name_hashs:
            path = os.path.join(root, data_file_name)
            if not os.path.exists(path) or not check_sha1(path, data_hash):
                downloaded_file_path = download(_get_repo_file_url(self._namespace,
                                                                   archive_file_name),
                                                path=root,
                                                sha1_hash=archive_hash)

                with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                    for member in zf.namelist():
                        filename = os.path.basename(member)
                        if filename:
                            dest = os.path.join(root, filename)
                            with zf.open(member) as source, \
                                    open(dest, 'wb') as target:
                                shutil.copyfileobj(source, target)
            paths.append(path)
        return paths

    def _get_data(self, segment):
        src_corpus_file_name, src_corpus_hash =\
            self._data_file[self._pair_key][segment + '_' + self._src_lang]
        tgt_corpus_file_name, tgt_corpus_hash =\
            self._data_file[self._pair_key][segment + '_' + self._tgt_lang]
        return self._fetch_data_path([(src_corpus_file_name, src_corpus_hash),
                                      (tgt_corpus_file_name, tgt_corpus_hash)])

    @property
    def src_vocab(self):
        """Source Vocabulary of the Dataset.

        Returns
        -------
        src_vocab : Vocab
            Source vocabulary.
        """
        if self._src_vocab is None:
            src_vocab_file_name, src_vocab_hash = \
                self._data_file[self._pair_key]['vocab' + '_' + self._src_lang]
            [src_vocab_path] = self._fetch_data_path([(src_vocab_file_name, src_vocab_hash)])
            with io.open(src_vocab_path, 'r', encoding='utf-8') as in_file:
                self._src_vocab = Vocab.from_json(in_file.read())
        return self._src_vocab

    @property
    def tgt_vocab(self):
        """Target Vocabulary of the Dataset.

        Returns
        -------
        tgt_vocab : Vocab
            Target vocabulary.
        """
        if self._tgt_vocab is None:
            tgt_vocab_file_name, tgt_vocab_hash = \
                self._data_file[self._pair_key]['vocab' + '_' + self._tgt_lang]
            [tgt_vocab_path] = self._fetch_data_path([(tgt_vocab_file_name, tgt_vocab_hash)])
            with io.open(tgt_vocab_path, 'r', encoding='utf-8') as in_file:
                self._tgt_vocab = Vocab.from_json(in_file.read())
        return self._tgt_vocab


@register(segment=['train', 'val', 'test'])
class IWSLT2015(_TranslationDataset):
    """Preprocessed IWSLT English-Vietnamese Translation Dataset.

    We use the preprocessed version provided in https://nlp.stanford.edu/projects/nmt/

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    src_lang : str, default 'en'
        The source language. Option for source and target languages are 'en' <-> 'vi'
    tgt_lang : str, default 'vi'
        The target language. Option for source and target languages are 'en' <-> 'vi'
    root : str, default '~/.mxnet/datasets/iwslt2015'
        Path to temp folder for storing data.
    """
    def __init__(self, segment='train', src_lang='en', tgt_lang='vi',
                 root=os.path.join('~', '.mxnet', 'datasets', 'iwslt2015')):
        self._supported_segments = ['train', 'val', 'test']
        self._archive_file = {_get_pair_key('en', 'vi'):
                                  ('iwslt15.zip', '15a05df23caccb1db458fb3f9d156308b97a217b')}
        self._data_file = {_get_pair_key('en', 'vi'):
                               {'train_en': ('train.en',
                                             '675d16d057f2b6268fb294124b1646d311477325'),
                                'train_vi': ('train.vi',
                                             'bb6e21d4b02b286f2a570374b0bf22fb070589fd'),
                                'val_en': ('tst2012.en',
                                           'e381f782d637b8db827d7b4d8bb3494822ec935e'),
                                'val_vi': ('tst2012.vi',
                                           '4511988ce67591dc8bcdbb999314715f21e5a1e1'),
                                'test_en': ('tst2013.en',
                                            'd320db4c8127a85de81802f239a6e6b1af473c3d'),
                                'test_vi': ('tst2013.vi',
                                            'af212c48a68465ceada9263a049f2331f8af6290'),
                                'vocab_en': ('vocab.en.json',
                                             'b6f8e77a45f6dce648327409acd5d52b37a45d94'),
                                'vocab_vi' : ('vocab.vi.json',
                                              '9be11a9edd8219647754d04e0793d2d8c19dc852')}}
        super(IWSLT2015, self).__init__('iwslt2015', segment=segment, src_lang=src_lang,
                                        tgt_lang=tgt_lang, root=root)


@register(segment=['train', 'newtest2012', 'newtest2013', 'newtest2014', \
                   'newtest2015', 'newtest2016'])
class WMT2016BPE(_TranslationDataset):
    """Preprocessed Translation Corpus of the WMT2016 Evaluation Campaign.

    We use the preprocessing script in
    https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'newstest2012', 'newstest2013',
         'newstest2014', 'newstest2015', 'newstest2016' or their combinations
    src_lang : str, default 'en'
        The source language. Option for source and target languages are 'en' <-> 'de'
    tgt_lang : str, default 'de'
        The target language. Option for source and target languages are 'en' <-> 'de'
    root : str, default '~/.mxnet/datasets/wmt2016'
        Path to temp folder for storing data.
    """
    def __init__(self, segment='train', src_lang='en', tgt_lang='de',
                 root=os.path.join('~', '.mxnet', 'datasets', 'wmt2016')):
        self._supported_segments = ['train'] + ['newstest%d' % i for i in range(2012, 2017)]
        self._archive_file = {_get_pair_key('de', 'en'):
                                  ('wmt2016_de_en.zip',
                                   '8cf0dbf6a102381443a472bcf9f181299231b496')}
        self._data_file = {_get_pair_key('de', 'en'):
                               {'train_en': ('train.tok.clean.bpe.32000.en',
                                             '56f37cb4d68c2f83efd6a0c555275d1fe09f36b5'),
                                'train_de': ('train.tok.clean.bpe.32000.de',
                                             '58f30a0ba7f80a8840a5cf3deff3c147de7d3f68'),
                                'newstest2012_en': ('newstest2012.tok.bpe.32000.en',
                                                    '25ed9ad228a236f57f97bf81db1bb004bedb7f33'),
                                'newstest2012_de': ('newstest2012.tok.bpe.32000.de',
                                                    'bb5622831ceea1894966fa993ebcd882cc461943'),
                                'newstest2013_en': ('newstest2013.tok.bpe.32000.en',
                                                    'fa03fe189fe68cb25014c5e64096ac8daf2919fa'),
                                'newstest2013_de': ('newstest2013.tok.bpe.32000.de',
                                                    '7d10a884499d352c2fea6f1badafb40473737640'),
                                'newstest2014_en': ('newstest2014.tok.bpe.32000.en',
                                                    '7b8ea824021cc5291e6a54bb32a1fc27c2955588'),
                                'newstest2014_de': ('newstest2014.tok.bpe.32000.de',
                                                    'd84497d4c425fa4e9b2b6be4b62c763086410aad'),
                                'newstest2015_en': ('newstest2015.tok.bpe.32000.en',
                                                    'ca335076f67b2f9b98848f8abc2cd424386f2309'),
                                'newstest2015_de': ('newstest2015.tok.bpe.32000.de',
                                                    'e633a3fb74506eb498fcad654d82c9b1a0a347b3'),
                                'newstest2016_en': ('newstest2016.tok.bpe.32000.en',
                                                    '5a5e36a6285823035b642aef7c1a9ec218da59f7'),
                                'newstest2016_de': ('newstest2016.tok.bpe.32000.de',
                                                    '135a79acb6a4f8fad0cbf5f74a15d9c0b5bf8c73'),
                                'vocab_en': ('vocab.bpe.32000.json',
                                             '1c5aea0a77cad592c4e9c1136ec3b70ceeff4e8c'),
                                'vocab_de': ('vocab.bpe.32000.json',
                                             '1c5aea0a77cad592c4e9c1136ec3b70ceeff4e8c')}}
        super(WMT2016BPE, self).__init__('wmt2016', segment=segment, src_lang=src_lang,
                                         tgt_lang=tgt_lang,
                                         root=os.path.join(root, _get_pair_key(src_lang, tgt_lang)))
