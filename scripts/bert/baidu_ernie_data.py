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
"""Baidu ernie data, contains XNLI."""

__all__ = ['BaiduErnieXNLI']

import os
import sys
import tarfile
from gluonnlp.data.dataset import TSVDataset
from gluonnlp.data.registry import register
from gluonnlp.base import get_home_dir
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

_baidu_ernie_data_url = 'https://ernie.bj.bcebos.com/task_data.tgz'

class _BaiduErnieDataset(TSVDataset):
    def __init__(self, root, dataset_name, segment, **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        download_data_path = os.path.join(self._root, 'task_data.tgz')
        if not os.path.exists(download_data_path):
            urlretrieve(_baidu_ernie_data_url, download_data_path)
            tar_file = tarfile.open(download_data_path, mode='r:gz')
            tar_file.extractall(self._root)
        filename = os.path.join(self._root, 'task_data', dataset_name, '%s.tsv' % segment)
        super(_BaiduErnieDataset, self).__init__(filename, **kwargs)


@register(segment=['train', 'dev', 'test'])
class BaiduErnieXNLI(_BaiduErnieDataset):
    """ The XNLI dataset released from Baidu
    <https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>.
    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/baidu_ernie_task_data'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> xnli_dev = BaiduErnieXNLI('dev', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(xnli_dev)
    2490
    >>> len(xnli_dev[0])
    3
    >>> xnli_dev[0]
    ['他说，妈妈，我回来了。', '校车把他放下后，他立即给他妈妈打了电话。', 'neutral']
    >>> xnli_test = BaiduErnieXNLI('test', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(xnli_test)
    5010
    >>> len(xnli_test[0])
    2
    >>> xnli_test[0]
    ['嗯，我根本没想过，但是我很沮丧，最后我又和他说话了。', '我还没有和他再次谈论。']
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'baidu_ernie_data'),
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
