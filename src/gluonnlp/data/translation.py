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

__all__ = ['IWSLT2015', 'WMT2014', 'WMT2014BPE', 'WMT2016', 'WMT2016BPE']


import os
import zipfile
import shutil
import io

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
from mxnet.gluon.data import ArrayDataset

from .dataset import TextLineDataset
from ..vocab import Vocab
from .registry import register
from .utils import _get_home_dir


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
    root : str, default '$MXNET_HOME/datasets/iwslt2015'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', src_lang='en', tgt_lang='vi',
                 root=os.path.join(_get_home_dir(), 'datasets', 'iwslt2015')):
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


@register(segment=['train', 'newstest2009', 'newstest2010', 'newstest2011', \
                   'newstest2012', 'newstest2013', 'newstest2014'])
class WMT2014(_TranslationDataset):
    """Translation Corpus of the WMT2014 Evaluation Campaign.

    http://www.statmt.org/wmt14/translation-task.html

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'newstest2009', 'newstest2010',
        'newstest2011', 'newstest2012', 'newstest2013', 'newstest2014' or their combinations
    src_lang : str, default 'en'
        The source language. Option for source and target languages are 'en' <-> 'de'
    tgt_lang : str, default 'de'
        The target language. Option for source and target languages are 'en' <-> 'de'
    full : bool, default False
        By default, we use the "filtered test sets" while if full is True, we use the "cleaned test
        sets".
    root : str, default '$MXNET_HOME/datasets/wmt2014'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', src_lang='en', tgt_lang='de', full=False,
                 root=os.path.join(_get_home_dir(), 'datasets', 'wmt2014')):
        self._supported_segments = ['train'] + ['newstest%d' % i for i in range(2009, 2015)]
        self._archive_file = {_get_pair_key('de', 'en'):
                                  ('wmt2014_de_en-b0e0e703.zip',
                                   'b0e0e7036217ffa94f4b35a5b5d2a96a27f680a4')}
        self._data_file = {_get_pair_key('de', 'en'):
                               {'train_en': ('train.en',
                                             'cec2d4c5035df2a54094076348eaf37e8b588a9b'),
                                'train_de': ('train.de',
                                             '6348764640ffc40992e7de89a8c48d32a8bcf458'),
                                'newstest2009_en': ('newstest2009.en',
                                                    'f8623af2de682924f9841488427e81c430e3ce60'),
                                'newstest2009_de': ('newstest2009.de',
                                                    'dec03f14cb47e726ccb19bec80c645d4a996f8a9'),
                                'newstest2010_en': ('newstest2010.en',
                                                    '5966eb13bd7cc8855cc6b40f9797607e36e9cc80'),
                                'newstest2010_de': ('newstest2010.de',
                                                    'b9af0cb004fa6996eda246d0173c191693b26025'),
                                'newstest2011_en': ('newstest2011.en',
                                                    '2c1d9d077fdbfe9d0e052a6e08a85ee7959479ab'),
                                'newstest2011_de': ('newstest2011.de',
                                                    'efbded3d175a9d472aa5938fe22afcc55c6055ff'),
                                'newstest2012_en': ('newstest2012.en',
                                                    '52f05ae725be45ee4012c6e208cef13614abacf1'),
                                'newstest2012_de': ('newstest2012.de',
                                                    'd9fe32143b88e6fe770843e15ee442a69ff6752d'),
                                'newstest2013_en': ('newstest2013.en',
                                                    '5dca5d02cf40278d8586ee7d58d58215253156a9'),
                                'newstest2013_de': ('newstest2013.de',
                                                    'ddda1e7b3270cb68108858640bfb619c37ede2ab'),
                                'newstest2014_en': ('newstest2014.src.en',
                                                    '610c5bb4cc866ad04ab1f6f80d740e1f4435027c'),
                                'newstest2014_de': ('newstest2014.ref.de',
                                                    '03b02c7f60c8509ba9bb4c85295358f7c9f00d2d')}}
        if full:
            self._data_file[_get_pair_key('de', 'en')]['newstest2014_en'] = \
                ('newstest2014.full.en', '528742a3a9690995d031f49d1dbb704844684976')
            self._data_file[_get_pair_key('de', 'en')]['newstest2014_de'] = \
                ('newstest2014.full.de', '2374b6a28cecbd965b73a9acc35a425e1ed81963')
        else:
            if src_lang == 'de':
                self._data_file[_get_pair_key('de', 'en')]['newstest2014_en'] = \
                    ('newstest2014.ref.en', 'cf23229ec6db8b85f240618d2a245f69afebed1f')
                self._data_file[_get_pair_key('de', 'en')]['newstest2014_de'] = \
                    ('newstest2014.src.de', '791d644b1a031268ca19600b2734a63c7bfcecc4')
        super(WMT2014, self).__init__('wmt2014', segment=segment, src_lang=src_lang,
                                      tgt_lang=tgt_lang,
                                      root=os.path.join(root, _get_pair_key(src_lang, tgt_lang)))


@register(segment=['train', 'newstest2009', 'newstest2010', 'newstest2011', \
                   'newstest2012', 'newstest2013', 'newstest2014'])
class WMT2014BPE(_TranslationDataset):
    """Preprocessed Translation Corpus of the WMT2014 Evaluation Campaign.

    We preprocess the dataset by adapting
    https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'newstest2009', 'newstest2010',
        'newstest2011', 'newstest2012', 'newstest2013', 'newstest2014' or their combinations
    src_lang : str, default 'en'
        The source language. Option for source and target languages are 'en' <-> 'de'
    tgt_lang : str, default 'de'
        The target language. Option for source and target languages are 'en' <-> 'de'
    full : bool, default False
        In default, we use the test dataset in http://statmt.org/wmt14/test-filtered.tgz.
        When full is True, we use the test dataset in http://statmt.org/wmt14/test-full.tgz
    root : str, default '$MXNET_HOME/datasets/wmt2014'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', src_lang='en', tgt_lang='de', full=False,
                 root=os.path.join(_get_home_dir(), 'datasets', 'wmt2014')):
        self._supported_segments = ['train'] + ['newstest%d' % i for i in range(2009, 2015)]
        self._archive_file = {_get_pair_key('de', 'en'):
                                  ('wmt2014bpe_de_en-ace8f41c.zip',
                                   'ace8f41c22c0da8729ff15f40d416ebd16738979')}
        self._data_file = {_get_pair_key('de', 'en'):
                               {'train_en': ('train.tok.clean.bpe.32000.en',
                                             'e3f093b64468db7084035c9650d9eecb86a3db5f'),
                                'train_de': ('train.tok.clean.bpe.32000.de',
                                             '60703ad088706a3d9d1f3328889c6f4725a36cfb'),
                                'newstest2009_en': ('newstest2009.tok.bpe.32000.en',
                                                    '5678547f579528a8716298e895f886e3976085e1'),
                                'newstest2009_de': ('newstest2009.tok.bpe.32000.de',
                                                    '32caa69023eac1750a0036780f9d511d979aed2c'),
                                'newstest2010_en': ('newstest2010.tok.bpe.32000.en',
                                                    '813103f7b4b472cf213fe3b2c3439e267dbc4afb'),
                                'newstest2010_de': ('newstest2010.tok.bpe.32000.de',
                                                    '972076a897ecbc7a3acb639961241b33fd58a374'),
                                'newstest2011_en': ('newstest2011.tok.bpe.32000.en',
                                                    'c3de2d72d5e7bdbe848839c55c284fece90464ce'),
                                'newstest2011_de': ('newstest2011.tok.bpe.32000.de',
                                                    '7a8722aeedacd99f1aa8dffb6d8d072430048011'),
                                'newstest2012_en': ('newstest2012.tok.bpe.32000.en',
                                                    '876ad3c72e33d8e1ed14f5362f97c771ce6a9c7f'),
                                'newstest2012_de': ('newstest2012.tok.bpe.32000.de',
                                                    '57467fcba8442164d058a05eaf642a1da1d92c13'),
                                'newstest2013_en': ('newstest2013.tok.bpe.32000.en',
                                                    'de06a155c3224674b2434f3ff3b2c4a4a293d238'),
                                'newstest2013_de': ('newstest2013.tok.bpe.32000.de',
                                                    '094084989128dd091a2fe2a5818a86bc99ecc0e7'),
                                'newstest2014_en': ('newstest2014.tok.bpe.32000.src.en',
                                                    '347cf4d3d5c3c46ca1220247d22c07aa90092bd9'),
                                'newstest2014_de': ('newstest2014.tok.bpe.32000.ref.de',
                                                    'f66b80a0c460c524ec42731e527c54aab5507a66'),
                                'vocab_en': ('vocab.bpe.32000.json',
                                             '71413f497ce3a0fa691c55277f367e5d672b27ee'),
                                'vocab_de': ('vocab.bpe.32000.json',
                                             '71413f497ce3a0fa691c55277f367e5d672b27ee')}}
        if full:
            self._data_file[_get_pair_key('de', 'en')]['newstest2014_en'] = \
                ('newstest2014.tok.bpe.32000.full.en', '6c398b61641cd39f186b417c54b171876563193f')
            self._data_file[_get_pair_key('de', 'en')]['newstest2014_de'] = \
                ('newstest2014.tok.bpe.32000.full.de', 'b890a8dfc2146dde570fcbcb42e4157292e95251')
        else:
            if src_lang == 'de':
                self._data_file[_get_pair_key('de', 'en')]['newstest2014_en'] = \
                    ('newstest2014.tok.bpe.32000.ref.en',
                     'cd416085db722bf07cbba4ff29942fe94e966023')
                self._data_file[_get_pair_key('de', 'en')]['newstest2014_de'] = \
                    ('newstest2014.tok.bpe.32000.src.de',
                     '9274d31f92141933f29a405753d5fae051fa5725')
        super(WMT2014BPE, self).__init__('wmt2014', segment=segment, src_lang=src_lang,
                                         tgt_lang=tgt_lang,
                                         root=os.path.join(root, _get_pair_key(src_lang, tgt_lang)))


@register(segment=['train', 'newstest2012', 'newstest2013', 'newstest2014', \
                   'newstest2015', 'newstest2016'])
class WMT2016(_TranslationDataset):
    """Translation Corpus of the WMT2016 Evaluation Campaign.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'newstest2012', 'newstest2013',
        'newstest2014', 'newstest2015', 'newstest2016' or their combinations
    src_lang : str, default 'en'
        The source language. Option for source and target languages are 'en' <-> 'de'
    tgt_lang : str, default 'de'
        The target language. Option for source and target languages are 'en' <-> 'de'
    root : str, default '$MXNET_HOME/datasets/wmt2016'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', src_lang='en', tgt_lang='de',
                 root=os.path.join(_get_home_dir(), 'datasets', 'wmt2016')):
        self._supported_segments = ['train'] + ['newstest%d' % i for i in range(2012, 2017)]
        self._archive_file = {_get_pair_key('de', 'en'):
                                  ('wmt2016_de_en-88767407.zip',
                                   '887674077b951ce949fe3e597086b826bd7574d8')}
        self._data_file = {_get_pair_key('de', 'en'):
                               {'train_en': ('train.en',
                                             '1be6d00c255c57183305276c5de60771e201d3b0'),
                                'train_de': ('train.de',
                                             '4eec608b8486bfb65b61bda237b0c9b3c0f66f17'),
                                'newstest2012_en': ('newstest2012.en',
                                                    '52f05ae725be45ee4012c6e208cef13614abacf1'),
                                'newstest2012_de': ('newstest2012.de',
                                                    'd9fe32143b88e6fe770843e15ee442a69ff6752d'),
                                'newstest2013_en': ('newstest2013.en',
                                                    '5dca5d02cf40278d8586ee7d58d58215253156a9'),
                                'newstest2013_de': ('newstest2013.de',
                                                    'ddda1e7b3270cb68108858640bfb619c37ede2ab'),
                                'newstest2014_en': ('newstest2014.en',
                                                    '528742a3a9690995d031f49d1dbb704844684976'),
                                'newstest2014_de': ('newstest2014.de',
                                                    '2374b6a28cecbd965b73a9acc35a425e1ed81963'),
                                'newstest2015_en': ('newstest2015.en',
                                                    'bf90439b209a496128995c4b948ad757979d0756'),
                                'newstest2015_de': ('newstest2015.de',
                                                    'd69ac825fe3d5796b4990b969ad71903a38a0cd1'),
                                'newstest2016_en': ('newstest2016.en',
                                                    'a99c145d5214eb1645b56d21b02a541fbe7eb3c2'),
                                'newstest2016_de': ('newstest2016.de',
                                                    'fcdd3104f21eb4b9c49ba8ddef46d9b2d472b3fe')}}
        super(WMT2016, self).__init__('wmt2016', segment=segment, src_lang=src_lang,
                                      tgt_lang=tgt_lang,
                                      root=os.path.join(root, _get_pair_key(src_lang, tgt_lang)))


@register(segment=['train', 'newstest2012', 'newstest2013', 'newstest2014', \
                   'newstest2015', 'newstest2016'])
class WMT2016BPE(_TranslationDataset):
    """Preprocessed Translation Corpus of the WMT2016 Evaluation Campaign.

    We use the preprocessing script in
    https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'newstest2012', 'newstest2013',
        'newstest2014', 'newstest2015', 'newstest2016' or their combinations
    src_lang : str, default 'en'
        The source language. Option for source and target languages are 'en' <-> 'de'
    tgt_lang : str, default 'de'
        The target language. Option for source and target languages are 'en' <-> 'de'
    root : str, default '$MXNET_HOME/datasets/wmt2016'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', src_lang='en', tgt_lang='de',
                 root=os.path.join(_get_home_dir(), 'datasets', 'wmt2016')):
        self._supported_segments = ['train'] + ['newstest%d' % i for i in range(2012, 2017)]
        self._archive_file = {_get_pair_key('de', 'en'):
                                  ('wmt2016bpe_de_en-8cf0dbf6.zip',
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
