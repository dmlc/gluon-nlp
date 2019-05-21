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

__all__ = ['BaiduErnieXNLI']

import os
import sys
import tarfile
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from gluonnlp.data.dataset import TSVDataset
from gluonnlp.data.registry import register
from gluonnlp.base import get_home_dir

_baidu_ernie_data_url = 'https://ernie.bj.bcebos.com/task_data.tgz'

class _BaiduErnieDataset(TSVDataset):
    def __init__(self, root, dataset_name, segment, **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        download_data_path = os.path.join(root, 'task_data.tgz')
        if not os.path.exists(download_data_path):
            urlretrieve(_baidu_ernie_data_url, download_data_path)
            tar_file = tarfile.open(download_data_path, mode='r:gz')
            tar_file.extractall(root)
        filename = os.path.join(self._root, 'task_data', dataset_name, '%s.tsv' % segment)
        super(_BaiduErnieDataset, self).__init__(filename, **kwargs)


@register(segment=['train', 'dev', 'test'])
class BaiduErnieXNLI(_BaiduErnieDataset):
    """ XNLI dataset

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/xnli'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> cola_dev = gluonnlp.data.BaiduErnieXNLI('dev', root='./datasets/xnli')
    -etc-
    >>> len(cola_dev)
    1043
    >>> len(cola_dev[0])
    2
    >>> cola_dev[0]
    ['The sailors rode the breeze clear of the rocks.', '1']
    >>> cola_test = gluonnlp.data.BaiduErnieXNLI('test', root='./datasets/xnli')
    -etc-
    >>> len(cola_test)
    1063
    >>> len(cola_test[0])
    1
    >>> cola_test[0]
    ['Bill whistled past the house.']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets'),
                 return_all_fields=False):
        A_IDX, B_IDX, LABEL_IDX = 0, 1, 2
        if segment in ['train', 'dev']:
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(BaiduErnieXNLI, self).__init__(root, 'xnli', segment,
                                       num_discard_samples=num_discard_samples,
                                       field_indices=field_indices)

