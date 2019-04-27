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

__all__ = ['GlueCoLA', 'GlueSST2', 'GlueSTSB', 'GlueQQP']

import codecs
import glob
import gzip
import zipfile
import io
import os
import shutil
import tarfile

from .dataset import TSVDataset
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from .. import _constants as C
from .registry import register
from ..base import get_home_dir

_glue_s3_uri = 's3://apache-mxnet/gluon/dataset/Glue/'

class _GlueDataset(TSVDataset):
    def __init__(self, root, data_file, **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        segment, zip_hash, data_hash = data_file
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
    """CoNLL2000 Part-of-speech (POS) tagging and chunking joint task dataset.

    Each sample has three fields: word, POS tag, chunk label.

    From
    https://www.clips.uantwerpen.be/cola2000/chunking/

    Parameters
    ----------
    segment : {'train', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/cola2000'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> cola = gluonnlp.data.GlueCoLA('test', root='./datasets/cola')
    -etc-
    >>> len(cola)
    1063
    >>> len(cola[0])
    1
    >>> cola[0][0]
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
            num_discard_samples=num_discard_samples, field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/CoLA'

@register(segment=['train', 'dev', 'test'])
class GlueSST2(_GlueDataset):
    """CoNLL2000 Part-of-speech (POS) tagging and chunking joint task dataset.

    Each sample has three fields: word, POS tag, chunk label.

    From
    https://www.clips.uantwerpen.be/cola2000/chunking/

    Parameters
    ----------
    segment : {'train', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/cola2000'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> cola = gluonnlp.data.GlueCoLA('test', root='./datasets/cola')
    -etc-
    >>> len(cola)
    1063
    >>> len(cola[0])
    1
    >>> cola[0][0]
    ['Bill whistled past the house.']
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
            num_discard_samples=num_discard_samples, field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/SST-2'

@register(segment=['train', 'dev', 'test'])
class GlueSTSB(_GlueDataset):
    """CoNLL2000 Part-of-speech (POS) tagging and chunking joint task dataset.

    Each sample has three fields: word, POS tag, chunk label.

    From
    https://www.clips.uantwerpen.be/cola2000/chunking/

    Parameters
    ----------
    segment : {'train', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/cola2000'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> cola = gluonnlp.data.GlueCoLA('test', root='./datasets/cola')
    -etc-
    >>> len(cola)
    1063
    >>> len(cola[0])
    1
    >>> cola[0][0]
    ['Bill whistled past the house.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_sst'),
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
            num_discard_samples=num_discard_samples, field_indices=field_indices)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/STS-B'

@register(segment=['train', 'dev', 'test'])
class GlueQQP(_GlueDataset):
    """CoNLL2000 Part-of-speech (POS) tagging and chunking joint task dataset.

    Each sample has three fields: word, POS tag, chunk label.

    From
    https://www.clips.uantwerpen.be/cola2000/chunking/

    Parameters
    ----------
    segment : {'train', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/cola2000'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> cola = gluonnlp.data.GlueCoLA('test', root='./datasets/cola')
    -etc-
    >>> len(cola)
    1063
    >>> len(cola[0])
    1
    >>> cola[0][0]
    ['Bill whistled past the house.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'glue_sst'),
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
            num_discard_samples=num_discard_samples, field_indices=field_indices,
            allow_missing=True)

    def _repo_dir(self):
        return 'gluon/dataset/GLUE/QQP'
