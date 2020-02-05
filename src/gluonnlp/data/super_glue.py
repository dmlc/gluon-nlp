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

# pylint: disable=line-too-long
"""SuperGLUEBenchmark corpora."""

__all__ = ['SuperGlueRTE', 'SuperGlueCB', 'SuperGlueWSC', 'SuperGlueWiC',
           'SuperGlueCOPA', 'SuperGlueMultiRC', 'SuperGlueBoolQ',
           'SuperGlueReCoRD', 'SuperGlueAXb', 'SuperGlueAXg']

import zipfile
import os
import re

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from .dataset import _JsonlDataset
from .registry import register
from ..base import get_home_dir


class _SuperGlueDataset(_JsonlDataset):
    def __init__(self, root, data_file):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        segment, zip_hash, data_hash = data_file
        self._root = root
        filename = os.path.join(self._root, '%s.jsonl' % segment)
        self._get_data(segment, zip_hash, data_hash, filename)
        super(_SuperGlueDataset, self).__init__(filename)

    def _get_data(self, segment, zip_hash, data_hash, filename):
        data_filename = '%s-%s.zip' % (segment, data_hash[:8])
        if not os.path.exists(filename) or not check_sha1(filename, data_hash):
            download(_get_repo_file_url(self._repo_dir(), data_filename),
                     path=self._root, sha1_hash=zip_hash)
            # unzip
            downloaded_path = os.path.join(self._root, data_filename)
            with zipfile.ZipFile(downloaded_path, 'r') as zf:
                # skip dir structures in the zip
                for zip_info in zf.infolist():
                    if zip_info.filename[-1] == '/':
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zf.extract(zip_info, self._root)

    def _repo_dir(self):
        raise NotImplementedError


@register(segment=['train', 'val', 'test'])
class SuperGlueRTE(_SuperGlueDataset):
    """The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual
    entailment challenges (RTE1, RTE2, RTE3 and RTE5).

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_rte"
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> rte_val = gluonnlp.data.SuperGlueRTE('val', root='./datasets/rte')
    -etc-
    >>> len(rte_val)
    277
    >>> sorted(rte_val[0].items())
    [('hypothesis', 'Christopher Reeve had an accident.'), ('idx', 0), ('label', 'not_entailment'), ('premise', 'Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation.')]
    >>> rte_test = gluonnlp.data.SuperGlueRTE('test', root='./datasets/rte')
    -etc-
    >>> len(rte_test)
    3000
    >>> sorted(rte_test[0].items())
    [('hypothesis', 'Shukla is related to Mangla.'), ('idx', 0), ('premise', "Mangla was summoned after Madhumita's sister Nidhi Shukla, who was the first witness in the case.")]

    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_rte')):
        self._segment = segment
        self._data_file = {'train': ('train', 'a4471b47b23f6d8bc2e89b2ccdcf9a3a987c69a1',
                                     '01ebec38ff3d2fdd849d3b33c2a83154d1476690'),
                           'val': ('val', '17f23360f77f04d03aee6c42a27a61a6378f1fd9',
                                   '410f8607d9fc46572c03f5488387327b33589069'),
                           'test': ('test', 'ef2de5f8351ef80036c4aeff9f3b46106b4f2835',
                                    '69f9d9b4089d0db5f0605eeaebc1c7abc044336b')}
        data_file = self._data_file[segment]

        super(SuperGlueRTE, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/RTE'


@register(segment=['train', 'val', 'test'])
class SuperGlueCB(_SuperGlueDataset):
    """The CommitmentBank (CB) is a corpus of short texts in which at least one sentence
    contains an embedded clause.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_cb"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> cb_val = gluonnlp.data.SuperGlueCB('val', root='./datasets/cb')
    -etc-
    >>> len(cb_val)
    56
    >>> sorted(cb_val[0].items())
    [('hypothesis', 'Valence was helping'), ('idx', 0), ('label', 'contradiction'), ('premise', "Valence the void-brain, Valence the virtuous valet. Why couldn't the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping?")]
    >>> cb_test = gluonnlp.data.SuperGlueCB('test', root='./datasets/cb')
    -etc-
    >>> len(cb_test)
    250
    >>> sorted(cb_test[0].items())
    [('hypothesis', 'Polly was not an experienced ocean sailor'), ('idx', 0), ('premise', 'Polly had to think quickly. They were still close enough to shore for him to return her to the police if she admitted she was not an experienced ocean sailor.')]
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_cb')):
        self._segment = segment
        self._data_file = {'train': ('train', '0b27cbdbbcdf2ba82da2f760e3ab40ed694bd2b9',
                                     '193bdb772d2fe77244e5a56b4d7ac298879ec529'),
                           'val': ('val', 'e1f9dc77327eba953eb41d5f9b402127d6954ae0',
                                   'd286ac7c9f722c2b660e764ec3be11bc1e1895f8'),
                           'test': ('test', '008f9afdc868b38fdd9f989babe034a3ac35dd06',
                                    'cca70739162d54f3cd671829d009a1ab4fd8ec6a')}
        data_file = self._data_file[segment]

        super(SuperGlueCB, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/CB'


@register(segment=['train', 'val', 'test'])
class SuperGlueWSC(_SuperGlueDataset):
    """
    The Winograd Schema Challenge (WSC) is a co-reference resolution dataset.
    (Levesque et al., 2012)

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_wsc"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> wsc_val = gluonnlp.data.SuperGlueWSC('val', root='./datasets/wsc')
    -etc-
    >>> len(wsc_val)
    104
    >>> sorted(wsc_val[5].items())
    [('idx', 5), ('label', True), ('target', OrderedDict([('span2_index', 9), ('span1_index', 6), ('span1_text', 'The table'), ('span2_text', 'it')])), ('text', 'The large ball crashed right through the table because it was made of styrofoam.')]
    >>> wsc_test = gluonnlp.data.SuperGlueWSC('test', root='./datasets/wsc')
    -etc-
    >>> len(wsc_test)
    146
    >>> sorted(wsc_test[16].items())
    [('idx', 16), ('target', OrderedDict([('span1_text', 'life'), ('span1_index', 1), ('span2_text', 'it'), ('span2_index', 21)])), ('text', 'Your life is yours and yours alone, and if the pain outweighs the benefit, you should have the option to end it .')]
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_wsc')):
        self._segment = segment
        self._data_file = {'train': ('train', 'ed0fe96914cfe1ae8eb9978877273f6baed621cf',
                                     'fa978f6ad4b014b5f5282dee4b6fdfdaeeb0d0df'),
                           'val': ('val', 'cebec2f5f00baa686573ae901bb4d919ca5d3483',
                                   'ea2413e4e6f628f2bb011c44e1d8bae301375211'),
                           'test': ('test', '3313896f315e0cb2bb1f24f3baecec7fc93124de',
                                    'a47024aa81a5e7c9bc6e957b36c97f1d1b5da2fd')}
        data_file = self._data_file[segment]

        super(SuperGlueWSC, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/WSC'


@register(segment=['train', 'val', 'test'])
class SuperGlueWiC(_SuperGlueDataset):
    """
    The Word-in-Context (WiC) is a word sense disambiguation dataset cast as binary classification
    of sentence pairs. (Pilehvar and Camacho-Collados, 2019)

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_wic"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> wic_val = gluonnlp.data.SuperGlueWiC('val', root='./datasets/wic')
    -etc-
    >>> len(wic_val)
    638
    >>> sorted(wic_val[3].items())
    [('end1', 31), ('end2', 35), ('idx', 3), ('label', True), ('sentence1', 'She gave her hair a quick brush.'), ('sentence2', 'The dentist recommended two brushes a day.'), ('start1', 26), ('start2', 28), ('version', 1.1), ('word', 'brush')]
    >>> wic_test = gluonnlp.data.SuperGlueWiC('test', root='./datasets/wic')
    -etc-
    >>> len(wic_test)
    1400
    >>> sorted(wic_test[0].items())
    [('end1', 46), ('end2', 22), ('idx', 0), ('sentence1', 'The smell of fried onions makes my mouth water.'), ('sentence2', 'His eyes were watering.'), ('start1', 41), ('start2', 14), ('version', 1.1), ('word', 'water')]
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_wic')):
        self._segment = segment
        self._data_file = {'train': ('train', 'ec1e265bbdcde1d8da0b56948ed30d86874b1f12',
                                     '831a58c553def448e1b1d0a8a36e2b987c81bc9c'),
                           'val': ('val', '2046c43e614d98d538a03924335daae7881f77cf',
                                   '73b71136a2dc2eeb3be7ab455a08f20b8dbe7526'),
                           'test': ('test', '77af78a49aac602b7bbf080a03b644167b781ba9',
                                    '1be93932d46c8f8dc665eb7af6703c56ca1b1e08')}
        data_file = self._data_file[segment]

        super(SuperGlueWiC, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/WiC'


@register(segment=['train', 'val', 'test'])
class SuperGlueCOPA(_SuperGlueDataset):
    """
    The Choice of Plausible Alternatives (COPA) is a causal reasoning dataset.
    (Roemmele et al., 2011)

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_copa"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> copa_val = gluonnlp.data.SuperGlueCOPA('val', root='./datasets/copa')
    -etc-
    >>> len(copa_val)
    100
    >>> sorted(copa_val[0].items())
    [('choice1', 'The toilet filled with water.'), ('choice2', 'Water flowed from the spout.'), ('idx', 0), ('label', 1), ('premise', 'The man turned on the faucet.'), ('question', 'effect')]
    >>> copa_test = gluonnlp.data.SuperGlueCOPA('test', root='./datasets/copa')
    -etc-
    >>> len(copa_test)
    500
    >>> sorted(copa_test[0].items())
    [('choice1', 'It was fragile.'), ('choice2', 'It was small.'), ('idx', 0), ('premise', 'The item was packaged in bubble wrap.'), ('question', 'cause')]
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_copa')):
        self._segment = segment
        self._data_file = {'train': ('train', '96d20163fa8371e2676a50469d186643a07c4e7b',
                                     '5bb9c8df7b165e831613c8606a20cbe5c9622cc3'),
                           'val': ('val', 'acc13ad855a1d2750a3b746fb0cfe3ca6e8b6615',
                                   'c8b908d880ffaf69bd897d6f2a1f23b8c3a732d4'),
                           'test': ('test', '89347d7884e71b49dd73c6bcc317aef64bb1bac8',
                                    '735f39f3d31409d83b16e56ad8aed7725ef5ddd5')}
        data_file = self._data_file[segment]

        super(SuperGlueCOPA, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/COPA'


@register(segment=['train', 'val', 'test'])
class SuperGlueMultiRC(_SuperGlueDataset):
    """
    Multi-Sentence Reading Comprehension (MultiRC) is a QA dataset.
    (Khashabi et al., 2018)

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_multirc"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> multirc_val = gluonnlp.data.SuperGlueMultiRC('val', root='./datasets/multirc')
    -etc-
    >>> len(multirc_val)
    83
    >>> sorted(multirc_val[0].keys())
    ['questions', 'text']
    >>> len(multirc_val[0]['text'])
    12
    >>> len(multirc_val[0]['questions'])
    13
    >>> sorted(multirc_val[0]['questions'][0].keys())
    ['answers', 'idx', 'multisent', 'question', 'sentences_used']
    >>> multirc_test = gluonnlp.data.SuperGlueMultiRC('test', root='./datasets/multirc')
    -etc-
    >>> len(multirc_test)
    166
    >>> sorted(multirc_test[0].keys())
    ['questions', 'text']
    >>> len(multirc_test[0]['text'])
    14
    >>> len(multirc_test[0]['questions'])
    14
    >>> sorted(multirc_test[0]['questions'][0].keys())
    ['answers', 'idx', 'multisent', 'question', 'sentences_used']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_multirc')):
        self._segment = segment
        self._data_file = {'train': ('train', '28d908566004fb84ff81828db8955f86fb771929',
                                     '2ef471a038f0b8116bf056da6440f290be7ab96e'),
                           'val': ('val', 'af93161bb987fbafe68111bce87dece4472b4ca0',
                                   '2364ed153f4f4e8cadde78680229a8544ba427db'),
                           'test': ('test', 'eabf1e8b426a8370cd3755a99412c7871a47ffa4',
                                    'd6d1107520d535332969ffe5f5b9bd7af2a33072')}
        data_file = self._data_file[segment]

        super(SuperGlueMultiRC, self).__init__(root, data_file)

    def _read_samples(self, samples):
        for i, sample in enumerate(samples):
            paragraph = dict()
            text = sample['paragraph']['text']
            sentences = self._split_text(text)
            paragraph['text'] = sentences
            paragraph['questions'] = sample['paragraph']['questions']
            samples[i] = paragraph
        return samples

    def _split_text(self, text):
        text = re.sub(r'<b>Sent .{1,2}: </b>', '', text)
        text = text.split('<br>')
        sents = [s for s in text if len(s) > 0]
        return sents

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/MultiRC'


@register(segment=['train', 'val', 'test'])
class SuperGlueBoolQ(_SuperGlueDataset):
    """
    Boolean Questions (BoolQ) is a QA dataset where each example consists of a short
    passage and a yes/no question about it.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_boolq"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> boolq_val = gluonnlp.data.SuperGlueBoolQ('val', root='./datasets/boolq')
    -etc-
    >>> len(boolq_val)
    3270
    >>> sorted(boolq_val[0].keys())
    ['idx', 'label', 'passage', 'question']
    >>> boolq_test = gluonnlp.data.SuperGlueBoolQ('test', root='./datasets/boolq')
    -etc-
    >>> len(boolq_test)
    3245
    >>> sorted(boolq_test[0].keys())
    ['idx', 'passage', 'question']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_boolq')):
        self._segment = segment
        self._data_file = {'train': ('train', '89507ff3015c3212b72318fb932cfb6d4e8417ef',
                                     'd5be523290f49fc0f21f4375900451fb803817c0'),
                           'val': ('val', 'fd39562fc2c9d0b2b8289d02a8cf82aa151d0ad4',
                                   '9b09ece2b1974e4da20f0173454ba82ff8ee1710'),
                           'test': ('test', 'a805d4bd03112366d548473a6848601c042667d3',
                                    '98c308620c6d6c0768ba093858c92e5a5550ce9b')}
        data_file = self._data_file[segment]

        super(SuperGlueBoolQ, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/BoolQ'


@register(segment=['train', 'val', 'test'])
class SuperGlueReCoRD(_SuperGlueDataset):
    """
    Reading Comprehension with Commonsense Reasoning Dataset (ReCoRD) is a multiple-choice
    QA dataset.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    root : str, default "$MXNET_HOME/datasets/superglue_record"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> record_val = gluonnlp.data.SuperGlueReCoRD('val', root='./datasets/record')
    -etc-
    >>> len(record_val)
    7481
    >>> sorted(record_val[0].keys())
    ['idx', 'passage', 'qas', 'source']
    >>> record_test = gluonnlp.data.SuperGlueReCoRD('test', root='./datasets/record')
    -etc-
    >>> len(record_test)
    7484
    >>> sorted(record_test[0].keys())
    ['idx', 'passage', 'qas', 'source']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_record')):
        self._segment = segment
        self._data_file = {'train': ('train', '047282c912535c9a3bcea519935fde882feb619d',
                                     '65592074cefde2ecd1b27ce7b35eb8beb86c691a'),
                           'val': ('val', '442d8470bff2c9295231cd10262a7abf401edc64',
                                   '9d1850e4dfe2eca3b71bfea191d5f4b412c65309'),
                           'test': ('test', 'fc639a18fa87befdc52f14c1092fb40475bf50d0',
                                    'b79b22f54b5a49f98fecd05751b122ccc6947c81')}
        data_file = self._data_file[segment]

        super(SuperGlueReCoRD, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/ReCoRD'


class SuperGlueAXb(_SuperGlueDataset):
    """
    The Broadcoverage Diagnostics (AX-b) is a diagnostics dataset labeled closely to
    the schema of MultiNLI.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    root : str, default "$MXNET_HOME/datasets/superglue_ax_b"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> ax_b = gluonnlp.data.SuperGlueAXb(root='./datasets/ax_b')
    -etc-
    >>> len(ax_b)
    1104
    >>> sorted(ax_b[0].items())
    [('idx', '0'), ('label', 'not_entailment'), ('logic', 'Negation'), ('sentence1', 'The cat sat on the mat.'), ('sentence2', 'The cat did not sit on the mat.')]
    """
    def __init__(self, root=os.path.join(get_home_dir(), 'datasets', 'superglue_ax_b')):
        data_file = ('AX-b', '398c5a376eb436f790723cd217ac040334140000',
                     '50fd8ac409897b652daa4b246917097c3c394bc8')

        super(SuperGlueAXb, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/AX-b'


class SuperGlueAXg(_SuperGlueDataset):
    """
    The Winogender Schema Diagnostics (AX-g) is a diagnostics dataset labeled closely to
    the schema of MultiNLI.

    From
    https://super.gluebenchmark.com/tasks

    Parameters
    ----------
    root : str, default "$MXNET_HOME/datasets/superglue_ax_g"
        Path to temp folder from storing data.
        MXNET_HOME defaults to '~/.mxnet'

    Examples
    --------
    >>> ax_g = gluonnlp.data.SuperGlueAXg(root='./datasets/ax_g')
    -etc-
    >>> len(ax_g)
    356
    >>> sorted(ax_g[0].items())
    [('hypothesis', 'The accountant sought help filing taxes.'), ('idx', 0), ('label', 'not_entailment'), ('pair_id', 551646), ('premise', 'The taxpayer met with the accountant to get help filing his taxes.')]
    """
    def __init__(self, root=os.path.join(get_home_dir(), 'datasets', 'superglue_ax_g')):
        data_file = ('AX-g', 'd8c92498496854807dfeacd344eddf466d7f468a',
                     '8a8cbfe00fd88776a2a2f20b477e5b0c6cc8ebae')

        super(SuperGlueAXg, self).__init__(root, data_file)

    def _read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/AX-g'
