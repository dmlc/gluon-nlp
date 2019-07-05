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
"""GLUEBenchmark corpora."""

__all__ = ['GlueCoLA', 'GlueSST2', 'GlueSTSB', 'GlueQQP', 'GlueRTE', 'GlueMNLI',
           'GlueQNLI', 'GlueWNLI', 'GlueMRPC']

import zipfile
import os
import io

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from .dataset import TSVDataset
from .registry import register
from ..base import get_home_dir


class _GlueDataset(TSVDataset):
    def __init__(self, root, data_file, **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        segment, zip_hash, data_hash = data_file
        self._root = root
        filename = os.path.join(self._root, '%s.tsv' % segment)
        self._get_data(segment, zip_hash, data_hash, filename)
        super(_GlueDataset, self).__init__(filename, **kwargs)

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

@register(segment=['train', 'dev', 'test'])
class GlueCoLA(_GlueDataset):
    """The Corpus of Linguistic Acceptability (Warstadt et al., 2018) consists of English
    acceptability judgments drawn from books and journal articles on linguistic theory.

    Each example is a sequence of words annotated with whether it is a grammatical
    English sentence.

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_cola'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> cola_dev = gluonnlp.data.GlueCoLA('dev', root='./datasets/cola')
    -etc-
    >>> len(cola_dev)
    1043
    >>> len(cola_dev[0])
    2
    >>> cola_dev[0]
    ['The sailors rode the breeze clear of the rocks.', '1']
    >>> cola_test = gluonnlp.data.GlueCoLA('test', root='./datasets/cola')
    -etc-
    >>> len(cola_test)
    1063
    >>> len(cola_test[0])
    1
    >>> cola_test[0]
    ['Bill whistled past the house.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_cola'),
                 return_all_fields=False):
        self._data_file = {'train': ('train', '662227ed4d98bb96b3495234b650e37826a5ef72',
                                     '7760a9c4b1fb05f6d003475cc7bb0d0118875190'),
                           'dev': ('dev', '6f3f5252b004eab187bf22ab5b0af31e739d3a3f',
                                   '30ece4de38e1929545c4154d4c71ad297c7f54b4'),
                           'test': ('test', 'b88180515ad041935793e74e3a76470b0c1b2c50',
                                    'f38b43d31bb06accf82a3d5b2fe434a752a74c9f')}
        data_file = self._data_file[segment]
        if segment in ['train', 'dev']:
            A_IDX, LABEL_IDX = 3, 1
            field_indices = [A_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 0
        elif segment == 'test':
            A_IDX = 1
            field_indices = [A_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(GlueCoLA, self).__init__(root, data_file,
                                       num_discard_samples=num_discard_samples,
                                       field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/CoLA'

@register(segment=['train', 'dev', 'test'])
class GlueSST2(_GlueDataset):
    """The Stanford Sentiment Treebank (Socher et al., 2013) consists of sentences from movie
    reviews and human annotations of their sentiment.

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_sst'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> sst_dev = gluonnlp.data.GlueSST2('dev', root='./datasets/sst')
    -etc-
    >>> len(sst_dev)
    872
    >>> len(sst_dev[0])
    2
    >>> sst_dev[0]
    ["it 's a charming and often affecting journey . ", '1']
    >>> sst_test = gluonnlp.data.GlueSST2('test', root='./datasets/sst')
    -etc-
    >>> len(sst_test)
    1821
    >>> len(sst_test[0])
    1
    >>> sst_test[0]
    ['uneasy mishmash of styles and genres .']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_sst'),
                 return_all_fields=False):
        self._data_file = {'train': ('train', 'bcde781bed5caa30d5e9a9d24e5c826965ed02a2',
                                     'ffbb67a55e27525e925b79fee110ca19585d70ca'),
                           'dev': ('dev', '85698e465ff6573fb80d0b34229c76df84cd766b',
                                   'e166f986cec68fd4cca0ae5ce5869b917f88a2fa'),
                           'test': ('test', 'efac1c275553ed78500e9b8d8629408f5f867b20',
                                    '3ce8041182bf82dbbbbfe13738b39d3c69722744')}
        data_file = self._data_file[segment]
        if segment in ['train', 'dev']:
            A_IDX, LABEL_IDX = 0, 1
            field_indices = [A_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            A_IDX = 1
            field_indices = [A_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(GlueSST2, self).__init__(root, data_file,
                                       num_discard_samples=num_discard_samples,
                                       field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/SST-2'

@register(segment=['train', 'dev', 'test'])
class GlueSTSB(_GlueDataset):
    """The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a collection of
    sentence pairs drawn from news headlines, video and image captions, and natural
    language inference data.

    Each pair is human-annotated with a similarity score from 1 to 5.

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_stsb'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> stsb_dev = gluonnlp.data.GlueSTSB('dev', root='./datasets/stsb')
    -etc-
    >>> len(stsb_dev)
    1500
    >>> len(stsb_dev[0])
    3
    >>> stsb_dev[0]
    ['A man with a hard hat is dancing.', 'A man wearing a hard hat is dancing.', '5.000']
    >>> stsb_test = gluonnlp.data.GlueSTSB('test', root='./datasets/stsb')
    -etc-
    >>> len(stsb_test)
    1379
    >>> len(stsb_test[0])
    2
    >>> stsb_test[0]
    ['A girl is styling her hair.', 'A girl is brushing her hair.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_stsb'),
                 return_all_fields=False):
        self._data_file = {'train': ('train', '9378bd341576810730a5c666ed03122e4c5ecc9f',
                                     '501e55248c6db2a3f416c75932a63693000a82bc'),
                           'dev': ('dev', '529c3e7c36d0807d88d0b2a5d4b954809ddd4228',
                                   'f8bcc33b01dfa2e9ba85601d0140020735b8eff3'),
                           'test': ('test', '6284872d6992d8ec6d96320af89c2f46ac076d18',
                                    '36553e5e2107b817257232350e95ff0f3271d844')}
        data_file = self._data_file[segment]
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 7, 8, 9
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            A_IDX, B_IDX, = 7, 8
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(GlueSTSB, self).__init__(root, data_file,
                                       num_discard_samples=num_discard_samples,
                                       field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/STS-B'

@register(segment=['train', 'dev', 'test'])
class GlueQQP(_GlueDataset):
    """The Quora Question Pairs dataset is a collection of question pairs from the community
    question-answering website Quora.

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_qqp'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> qqp_dev = gluonnlp.data.GlueQQP('dev', root='./datasets/qqp')
    -etc-
    >>> len(qqp_dev)
    40430
    >>> len(qqp_dev[0])
    3
    >>> qqp_dev[0]
    ['Why are African-Americans so beautiful?', 'Why are hispanics so beautiful?', '0']
    >>> qqp_test = gluonnlp.data.GlueQQP('test', root='./datasets/qqp')
    -etc-
    >>> len(qqp_test)
    390965
    >>> len(qqp_test[3])
    2
    >>> qqp_test[3]
    ['Is it safe to invest in social trade biz?', 'Is social trade geniune?']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_qqp'),
                 return_all_fields=False):
        self._data_file = {'train': ('train', '494f280d651f168ad96d6cd05f8d4ddc6be73ce9',
                                     '95c01e711ac8dbbda8f67f3a4291e583a72b6988'),
                           'dev': ('dev', '9957b60c4c62f9b98ec91b26a9d43529d2ee285d',
                                   '755e0bf2899b8ad315d4bd7d4c85ec51beee5ad0'),
                           'test': ('test', '1e325cc5dbeeb358f9429c619ebe974fc2d1a8ca',
                                    '0f50d1a62dd51fe932ba91be08238e47c3e2504a')}
        data_file = self._data_file[segment]
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            A_IDX, B_IDX, = 1, 2
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1
        # QQP may include broken samples
        super(GlueQQP, self).__init__(root, data_file,
                                      num_discard_samples=num_discard_samples,
                                      field_indices=field_indices, allow_missing=True)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/QQP'

@register(segment=['train', 'dev', 'test'])
class GlueRTE(_GlueDataset):
    """The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual
    entailment challenges (RTE1, RTE2, RTE3, and RTE5).

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_rte'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> rte_dev = gluonnlp.data.GlueRTE('dev', root='./datasets/rte')
    -etc-
    >>> len(rte_dev)
    277
    >>> len(rte_dev[0])
    3
    >>> rte_dev[0]
    ['Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation.', 'Christopher Reeve had an accident.', 'not_entailment']
    >>> rte_test = gluonnlp.data.GlueRTE('test', root='./datasets/rte')
    -etc-
    >>> len(rte_test)
    3000
    >>> len(rte_test[16])
    2
    >>> rte_test[16]
    ['United failed to progress beyond the group stages of the Champions League and trail in the Premiership title race, sparking rumours over its future.', 'United won the Champions League.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_rte'),
                 return_all_fields=False):
        self._data_file = {'train': ('train', 'a23b0633f4f4dfa866c672af2e94f7e07344888f',
                                     'ec2b246745bb5c9d92aee0800684c08902742730'),
                           'dev': ('dev', 'a6cde090d12a10744716304008cf33dd3f0dbfcb',
                                   'ade75e0673862dcac9c653efb9f59f51be2749aa'),
                           'test': ('test', '7e4e58a6fa80b1f05e603b4e220524be7976b488',
                                    'ddda5c967fb5a4934b429bb52aaa144e70900000')}
        data_file = self._data_file[segment]
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            A_IDX, B_IDX, = 1, 2
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1
        super(GlueRTE, self).__init__(root, data_file,
                                      num_discard_samples=num_discard_samples,
                                      field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/RTE'

@register(segment=['train', 'dev_matched', 'dev_mismatched',
                   'test_matched', 'test_mismatched'])
class GlueMNLI(_GlueDataset):
    """The Multi-Genre Natural Language Inference Corpus (Williams et al., 2018)
    is a crowdsourced collection of sentence pairs with textual entailment annotations.

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched'},
              default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_mnli'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> mnli_dev = gluonnlp.data.GlueMNLI('dev_matched', root='./datasets/mnli')
    -etc-
    >>> len(mnli_dev)
    9815
    >>> len(mnli_dev[0])
    3
    >>> mnli_dev[0]
    ['The new rights are nice enough', 'Everyone really likes the newest benefits ', 'neutral']
    >>> mnli_test = gluonnlp.data.GlueMNLI('test_matched', root='./datasets/mnli')
    -etc-
    >>> len(mnli_test)
    9796
    >>> len(mnli_test[0])
    2
    >>> mnli_test[0]
    ['Hierbas, ans seco, ans dulce, and frigola are just a few names worth keeping a look-out for.', 'Hierbas is a name worth looking out for.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_mnli'),
                 return_all_fields=False):
        self._data_file = {'train': ('train', 'aa235064ab3ce47d48caa17c553561d84fdf5bf2',
                                     '1e74055bc91e260323574bfe63186acb9420fa13'),
                           'dev_matched': ('dev_matched',
                                           '328cf527add50ee7bc20a862f97913800ba8a4b1',
                                           '7a38c5fb5ecc875f259e1d57662d58a984753b70'),
                           'dev_mismatched': ('dev_mismatched',
                                              '9c5d6c6d2e3a676bfa19d929b32e2f9f233585c5',
                                              '47470d91b594e767d80e5de2ef0be6a453c17be5'),
                           'test_matched': ('test_matched',
                                            '53877d9d554b6a6d402cc0e5f7e38366cd4f8e60',
                                            '00106769e11a43eac119975ad25c2de2c8d2dbe7'),
                           'test_mismatched': ('test_mismatched',
                                               '82b03d3cc9f4a59c74beab06c141bc0c5bf74a55',
                                               '5a31abf92f045f127dbb2e3d2e0ef8ddea04c237')}
        data_file = self._data_file[segment]
        if segment in ['train']:
            A_IDX, B_IDX, LABEL_IDX = 8, 9, 11
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment in ['dev_matched', 'dev_mismatched']:
            A_IDX, B_IDX, LABEL_IDX = 8, 9, 15
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment in ['test_matched', 'test_mismatched']:
            A_IDX, B_IDX, = 8, 9
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1
        super(GlueMNLI, self).__init__(root, data_file,
                                       num_discard_samples=num_discard_samples,
                                       field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/MNLI'

@register(segment=['train', 'dev', 'test'])
class GlueQNLI(_GlueDataset):
    r"""The Question-answering NLI dataset converted from Stanford Question Answering Dataset
    (Rajpurkar et al. 2016).

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_qnli'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> qnli_dev = gluonnlp.data.GlueQNLI('dev', root='./datasets/qnli')
    -etc-
    >>> len(qnli_dev)
    5732
    >>> len(qnli_dev[0])
    3
    >>> qnli_dev[0]
    ['Which NFL team represented the AFC at Super Bowl 50?', 'The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title.', 'entailment']
    >>> qnli_test = gluonnlp.data.GlueQNLI('test', root='./datasets/qnli')
    -etc-
    >>> len(qnli_test)
    5740
    >>> len(qnli_test[0])
    2
    >>> qnli_test[0]
    ['What seldom used term of a unit of force equal to 1000 pound s of force?', 'Other arcane units of force include the sthène, which is equivalent to 1000 N, and the kip, which is equivalent to 1000 lbf.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_qnli'),
                 return_all_fields=False):
        self._data_file = {'train': ('train', '95fae96fb1ffa6a2804192c9036d3435e63b48e8',
                                     'd90a84eb40c6ba32bc2b34284ceaa962c46f8753'),
                           'dev': ('dev', '5652b9d4d5c8d115c080bcf64101927ea2b3a1e0',
                                   'd14a61290301c2a9d26459c4cd036742e8591428'),
                           'test': ('test', '23dfb2f38adb14d3e792dbaecb7f5fd5dfa8db7e',
                                    'f3da1a2e471ebfee81d91574b42e0f5d39153c59')}
        data_file = self._data_file[segment]
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            A_IDX, B_IDX, = 1, 2
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1
        super(GlueQNLI, self).__init__(root, data_file,
                                       num_discard_samples=num_discard_samples,
                                       field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/QNLI'

@register(segment=['train', 'dev', 'test'])
class GlueWNLI(_GlueDataset):
    """The Winograd NLI dataset converted from the dataset in
    Winograd Schema Challenge (Levesque et al., 2011).

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_wnli'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> wnli_dev = gluonnlp.data.GlueWNLI('dev', root='./datasets/wnli')
    -etc-
    >>> len(wnli_dev)
    71
    >>> len(wnli_dev[0])
    3
    >>> wnli_dev[0]
    ['The drain is clogged with hair. It has to be cleaned.', 'The hair has to be cleaned.', '0']
    >>> wnli_test = gluonnlp.data.GlueWNLI('test', root='./datasets/wnli')
    -etc-
    >>> len(wnli_test)
    146
    >>> len(wnli_test[0])
    2
    >>> wnli_test[0]
    ['Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they came in sight.', 'Horses ran away when Maude and Dora came in sight.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_wnli'),
                 return_all_fields=False):
        self._data_file = {'train': ('train', '8db0004d0e58640751a9f2875dd66c8000504ddb',
                                     'b497281c1d848b619ea8fe427b3a6e4dc8e7fa92'),
                           'dev': ('dev', 'd54834960555073fb497cf2766edb77fb62c3646',
                                   '6bbdb866d0cccaac57c3a2505cf53103789b69a9'),
                           'test': ('test', '431e596a1c6627fb168e7741b3e32ef681da3c7b',
                                    '6ba8fcf3e5b451c101a3902fb4ba3fc1dea42e50')}
        data_file = self._data_file[segment]
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            A_IDX, B_IDX, = 1, 2
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1
        super(GlueWNLI, self).__init__(root, data_file,
                                       num_discard_samples=num_discard_samples,
                                       field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/WNLI'

@register(segment=['train', 'dev', 'test'])
class GlueMRPC(TSVDataset):
    """The Microsoft Research Paraphrase Corpus dataset.

    From
    https://gluebenchmark.com/tasks

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/glue_mrpc'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> mrpc_dev = gluonnlp.data.GlueMRPC('dev', root='./datasets/mrpc')
    -etc-
    >>> len(mrpc_dev)
    408
    >>> len(mrpc_dev[0])
    3
    >>> mrpc_dev[0]
    ["He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .", '" The foodservice pie business does not fit our long-term growth strategy .', '1']
    >>> mrpc_test = gluonnlp.data.GlueMRPC('test', root='./datasets/mrpc')
    -etc-
    >>> len(mrpc_test)
    1725
    >>> len(mrpc_test[0])
    2
    >>> mrpc_test[0]
    ["PCCW 's chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So .", 'Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So .']
    """
    def __init__(self,
                 segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_mrpc')):
        self._root = root
        assert segment in ['train', 'dev', 'test'], 'Unsupported segment: %s'%segment
        self._data_file = {'train': ('msr_paraphrase_train.txt',
                                     '716e0f67af962f08220b7e97d229b293077ef41f',
                                     '131675ffd3d2f04f286049d31cca506c8acba69e'),
                           'dev': ('msr_paraphrase_train.txt',
                                   '716e0f67af962f08220b7e97d229b293077ef41f',
                                   'e4486577c4cb2e5c2a3fd961eb24f03c623ea02d'),
                           'test': ('msr_paraphrase_test.txt',
                                    '4265196c15cf75620b0b592b8b921f543bda7e6c',
                                    '3602b2ca26cf574e84183c14d6c0901669ee2d0a')}

        self._generate(segment)
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX, LABEL_IDX = 3, 4, 0
        if segment == 'test':
            fields = [A_IDX, B_IDX]
        else:
            fields = [A_IDX, B_IDX, LABEL_IDX]
        super(GlueMRPC, self).__init__(
            path, num_discard_samples=1, field_indices=fields)

    def _repo_dir(self):
        return 'https://dl.fbaipublicfiles.com/senteval/senteval_data/'

    def _generate(self, segment):
        """Partition MRPC dataset into train, dev and test.
        Adapted from https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
        """
        # download raw data
        data_name = segment + '.tsv'
        raw_name, raw_hash, data_hash = self._data_file[segment]
        raw_path = os.path.join(self._root, raw_name)
        download(self._repo_dir() + raw_name, path=raw_path, sha1_hash=raw_hash)
        data_path = os.path.join(self._root, data_name)

        if segment == 'train' or segment == 'dev':
            if os.path.isfile(data_path) and check_sha1(data_path, data_hash):
                return

            # retrieve dev ids for train and dev set
            DEV_ID_URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc'
            DEV_ID_HASH = '506c7a1a5e0dd551ceec2f84070fa1a8c2bc4b41'
            dev_id_name = 'dev_ids.tsv'
            dev_id_path = os.path.join(self._root, dev_id_name)
            download(DEV_ID_URL, path=dev_id_path, sha1_hash=DEV_ID_HASH)

            # read dev data ids
            dev_ids = []
            with io.open(dev_id_path, encoding='utf8') as ids_fh:
                for row in ids_fh:
                    dev_ids.append(row.strip().split('\t'))

            # generate train and dev set
            train_path = os.path.join(self._root, 'train.tsv')
            dev_path = os.path.join(self._root, 'dev.tsv')
            with io.open(raw_path, encoding='utf8') as data_fh:
                with io.open(train_path, 'w', encoding='utf8') as train_fh:
                    with io.open(dev_path, 'w', encoding='utf8') as dev_fh:
                        header = data_fh.readline()
                        train_fh.write(header)
                        dev_fh.write(header)
                        for row in data_fh:
                            label, id1, id2, s1, s2 = row.strip().split('\t')
                            example = u'%s\t%s\t%s\t%s\t%s\n'%(label, id1, id2, s1, s2)
                            if [id1, id2] in dev_ids:
                                dev_fh.write(example)
                            else:
                                train_fh.write(example)
        else:
            # generate test set
            if os.path.isfile(data_path) and check_sha1(data_path, data_hash):
                return
            with io.open(raw_path, encoding='utf8') as data_fh:
                with io.open(data_path, 'w', encoding='utf8') as test_fh:
                    header = data_fh.readline()
                    test_fh.write(u'index\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
                    for idx, row in enumerate(data_fh):
                        label, id1, id2, s1, s2 = row.strip().split('\t')
                        test_fh.write(u'%d\t%s\t%s\t%s\t%s\n'%(idx, id1, id2, s1, s2))
