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
"""CoNLL format corpora."""

__all__ = ['CoNLL2000', 'CoNLL2001', 'CoNLL2002', 'CoNLL2004', 'UniversalDependencies21']

import codecs
import glob
import gzip
import io
import os
import shutil
import tarfile

from mxnet.gluon.data import SimpleDataset
from mxnet.gluon.utils import download, check_sha1

from .. import _constants as C
from .registry import register
from .utils import _get_home_dir


class _CoNLLSequenceTagging(SimpleDataset):
    def __init__(self, segment, root, has_comment=False):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._segment = segment
        self._root = root
        self._has_comment = has_comment
        super(_CoNLLSequenceTagging, self).__init__(self._read_data())

    def _get_data_file_hash(self):
        assert self._segment in self._data_file, \
                'Segment "{}" is not available. Options are: {}.'.format(self._segment,
                                                                         self._data_files.keys())
        return [self._data_file[self._segment]]

    def _get_data_archive_hash(self):
        return self._get_data_file_hash()[0]

    def _extract_archive(self):
        pass

    def _get_data(self):
        archive_file_name, archive_hash = self._get_data_archive_hash()
        paths = []
        for data_file_name, data_hash in self._get_data_file_hash():
            root = self._root
            path = os.path.join(root, data_file_name)
            if not os.path.exists(path) or not check_sha1(path, data_hash):
                download(self.base_url + archive_file_name,
                         path=root, sha1_hash=archive_hash)
                self._extract_archive()
            paths.append(path)
        return paths

    def _read_data(self):
        paths = self._get_data()
        results = []
        for path in paths:
            with gzip.open(path, 'r') if path.endswith('gz') else io.open(path, 'rb') as f:
                line_iter = codecs.getreader(self.codec)(io.BufferedReader(f))
                results.append(self._process_iter(line_iter))
        return list([x for field in item for x in field] for item in zip(*results))

    def _process_iter(self, line_iter):
        samples = []
        buf = []
        for line in line_iter:
            if not buf and line.startswith('#') and self._has_comment:
                continue
            line = line.split()
            if line:
                buf.append(line)
            elif buf:
                samples.append(tuple(map(list, zip(*buf))))
                buf = []
        if buf:
            samples.append(tuple(map(list, zip(*buf))))
        return samples


@register(segment=['train', 'test'])
class CoNLL2000(_CoNLLSequenceTagging):
    """CoNLL2000 Part-of-speech (POS) tagging and chunking joint task dataset.

    Each sample has three fields: word, POS tag, chunk label.

    From
    https://www.clips.uantwerpen.be/conll2000/chunking/

    Parameters
    ----------
    segment : {'train', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/conll2000'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train',
                 root=os.path.join(_get_home_dir(), 'datasets', 'conll2000')):
        self._data_file = {'train': ('train.txt.gz',
                                     '9f31cf936554cebf558d07cce923dca0b7f31864'),
                           'test': ('test.txt.gz',
                                    'dc57527f1f60eeafad03da51235185141152f849')}
        super(CoNLL2000, self).__init__(segment, root)

    base_url = 'http://www.clips.uantwerpen.be/conll2000/chunking/'
    codec = 'utf-8'


@register(segment=['train', 'testa', 'testb'], part=[1, 2, 3])
class CoNLL2001(_CoNLLSequenceTagging):
    """CoNLL2001 Clause Identification dataset.

    Each sample has four fields: word, POS tag, chunk label, clause tag.

    From
    https://www.clips.uantwerpen.be/conll2001/clauses/

    Parameters
    ----------
    part : int, {1, 2, 3}
        Part number of the dataset.
    segment : {'train', 'testa', 'testb'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/conll2001'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, part, segment='train',
                 root=os.path.join(_get_home_dir(), 'datasets', 'conll2001')):
        self._part = part
        self._data_file = [
            {'train': ('train1',
                       '115400d32437a86af85fbd549c1297775aec5996'),
             'testa': ('testa1',
                       '0fad761a9c3e0fece80550add3420554619bce66'),
             'testb': ('testb1',
                       'f1075e69b57a9c8e5e5de8496610469dcaaca511')},
            {'train': ('train2',
                       'd48cf110875e5999e20e72bc446c9dd19fdde618'),
             'testa': ('testa2',
                       '27262d3a45e6b08631d8c2c8d8c33cf7fd63db2c'),
             'testb': ('testb2',
                       'd8d0b5819ca5e275c25cec0287ffff8811e65321')},
            {'train': ('train3',
                       'c064ba4cb54f81a3d1e15d48cc990dee55a326bc'),
             'testa': ('testa3',
                       'c0c11cceb17bba8e0aaad0368d8b0b869c4959f7'),
             'testb': ('testb3',
                       'a37f3ca89eb4db08fc576f50161f6c2945302541')}
            ]
        super(CoNLL2001, self).__init__(segment, root)

    base_url = 'https://www.clips.uantwerpen.be/conll2001/clauses/data/'
    codec = 'utf-8'

    def _get_data_file_hash(self):
        assert self._part in [1, 2, 3], \
                'Part "{}" is not availble. Options are 1, 2, 3.'.format(self._part)
        available_segments = self._data_file[self._part-1].keys()
        assert self._segment in available_segments, \
                'Segment "{}" is not available. Options are: {}.'.format(self._segment,
                                                                         available_segments)
        return [self._data_file[self._part-1][self._segment]]


@register(segment=['train', 'testa', 'testb'], lang=['esp', 'ned'])
class CoNLL2002(_CoNLLSequenceTagging):
    """CoNLL2002 Named Entity Recognition (NER) task dataset.

    For 'esp', each sample has two fields: word, NER label.

    For 'ned', each sample has three fields: word, POS tag, NER label.

    From
    https://www.clips.uantwerpen.be/conll2002/ner/

    Parameters
    ----------
    lang : str, {'esp', 'ned'}
        Dataset language.
    segment : {'train', 'testa', 'testb'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/conll2002'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, lang, segment='train',
                 root=os.path.join(_get_home_dir(), 'datasets', 'conll2002')):
        self._lang = lang
        self._data_file = {
            'esp': {'train': ('esp.train.gz',
                              '2f25c8c1a724009f440af8bb3c03710f089dfe11'),
                    'testa': ('esp.testa.gz',
                              '1afd035a29419b1a9531308cae6157c624260693'),
                    'testb': ('esp.testb.gz',
                              'c6a16bcb0399bf212fec80d6049eaeffcdb58c1d')},
            'ned': {'train': ('ned.train.gz',
                              '4282015737b588efa13e6616222d238247a85c67'),
                    'testa': ('ned.testa.gz',
                              '7584cbf55692d3b0c133de6d7411ad04ae0e710a'),
                    'testb': ('ned.testb.gz',
                              '4d07c576f99aae8a305855a9cbf40163c0b8d84e')}}
        super(CoNLL2002, self).__init__(segment, root)

    base_url = 'https://www.clips.uantwerpen.be/conll2002/ner/data/'
    codec = 'latin-1'

    def _get_data_file_hash(self):
        assert self._lang in self._data_file, \
                'Language "{}" is not available. Options are "{}".'.format(self._lang,
                                                                           self._data_file.keys())
        available_segments = self._data_file[self._lang].keys()
        assert self._segment in available_segments, \
                'Segment "{}" is not available. Options are: {}.'.format(self._segment,
                                                                         available_segments)
        return [self._data_file[self._lang][self._segment]]


@register(segment=['train', 'dev', 'test'])
class CoNLL2004(_CoNLLSequenceTagging):
    """CoNLL2004 Semantic Role Labeling (SRL) task dataset.

    Each sample has seven or more fields: word, POS tag, chunk label, clause tag, NER label,
    target verbs, and sense labels (of variable number per sample).

    From
    http://www.cs.upc.edu/~srlconll/st04/st04.html

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/conll2004'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train',
                 root=os.path.join(_get_home_dir(), 'datasets', 'conll2004')):
        self._archive_file = ('conll04st-release.tar.gz',
                              '09ef957d908d34fa0abd745cbe43e279414f076c')
        self._data_file = {
            'word': {'train': ('words.train.gz',
                               '89ac63dcdcffc71601a224be6ada7f2e67c8e61f'),
                     'dev': ('words.dev.gz',
                             'c3e59d75ae6bbeb76ee78e52a7a7c6b52abc5b6f'),
                     'test': ('words.test.gz',
                              '61c7653732d83b51593ed29ae7ff45cd8277c8b5')},
            'synt': {'train': ('synt.train.pred.gz',
                               '43ed796f953dcf00db52ec593ed3377aa440d838'),
                     'dev': ('synt.dev.pred.gz',
                             'c098ca8a265fb67529c90eee5a93f6781ad87747'),
                     'test': ('synt.test.pred.gz',
                              '272c2856171f3e28e3512906ee07019bac90a6b2')},
            'ne': {'train': ('ne.train.pred.gz',
                             'd10e8b11b6b856efac978697af75cf582cac6e86'),
                   'dev': ('ne.dev.pred.gz',
                           '7883f76f28675d2a7247be527967b846494bbe2c'),
                   'test': ('ne.test.pred.gz',
                            'f1a52a58bb96e07e0288479a4a633476d8211963')},
            'props': {'train': ('props.train.gz',
                                'c67bb4546e9110ce39ce063624c7a0adf65ea795'),
                      'dev': ('props.dev.gz',
                              '7e232a4113d1a7e68b719a2781f09399ebf39956'),
                      'test': ('props.test.gz',
                               '639d54e24cebd7476b05c0efc0cbb019ebe52d8e')}}

        super(CoNLL2004, self).__init__(segment, root)

    base_url = 'http://www.cs.upc.edu/~srlconll/st04/'
    codec = 'utf-8'

    def _get_data_file_hash(self):
        available_segments = self._data_file['ne'].keys()
        assert self._segment in self._data_file['ne'], \
                'Segment "{}" is not available. Options are: {}'.format(self._segment,
                                                                        available_segments)
        return [self._data_file[part][self._segment] for part in ['word', 'synt', 'ne', 'props']]

    def _get_data_archive_hash(self):
        return self._archive_file

    def _extract_archive(self):
        archive_file_name, _ = self._get_data_archive_hash()
        root = self._root
        path = os.path.join(root, archive_file_name)
        with tarfile.open(path, 'r:gz') as tar:
            tar.extractall(path=root)
        for fn in glob.glob(os.path.join(root, 'conll04st-release', '*.gz')):
            shutil.copy(fn, root)
        shutil.rmtree(os.path.join(root, 'conll04st-release'), ignore_errors=True)


@register(segment=['train', 'dev', 'test'],
          lang=list(C.UD21_DATA_FILE_SHA1.keys()))
class UniversalDependencies21(_CoNLLSequenceTagging):
    """Universal dependencies tree banks.

    Each sample has 8 or more fields as described in
    http://universaldependencies.org/docs/format.html

    From
    https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2515

    Parameters
    ----------
    lang : str, default 'en'
        Dataset language.
    segment : str, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/ud2.1'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, lang='en', segment='train',
                 root=os.path.join(_get_home_dir(), 'datasets', 'ud2.1')):
        self._archive_file = ('ud-treebanks-v2.1.tgz',
                              '77657b897951e21d2eca6b29958e663964eb57ae')
        self._lang = lang
        self._data_file = C.UD21_DATA_FILE_SHA1

        super(UniversalDependencies21, self).__init__(segment, root, True)

    base_url = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2515/'
    codec = 'utf-8'

    def _get_data_file_hash(self):
        assert self._lang in self._data_file, \
                'Language "{}" is not available. Options are {}.'.format(self._lang,
                                                                         self._data_file.values())
        available_segments = self._data_file[self._lang].keys()
        assert self._segment in available_segments, \
                'Segment "{}" is not available for language "{}". ' \
                'Options are: {}.'.format(self._segment, self._lang, available_segments)
        return [self._data_file[self._lang][self._segment]]

    def _get_data_archive_hash(self):
        return self._archive_file

    def _extract_archive(self):
        archive_file_name, _ = self._get_data_archive_hash()
        root = self._root
        path = os.path.join(root, archive_file_name)
        with tarfile.open(path, 'r:gz') as tar:
            tar.extractall(path=root)
        for fn in glob.glob(os.path.join(root, 'ud-treebanks-v2.1', '*', '*.conllu')):
            shutil.copy(fn, root)
        for data_license in glob.glob(os.path.join(root, 'ud-treebanks-v2.1', '*', 'LICENSE.txt')):
            lang = os.path.dirname(data_license).split(os.path.sep)[-1]
            shutil.copy(data_license, os.path.join(root, '{}_LICENSE.txt'.format(lang)))
        shutil.rmtree(os.path.join(root, 'ud-treebanks-v2.1'), ignore_errors=True)
