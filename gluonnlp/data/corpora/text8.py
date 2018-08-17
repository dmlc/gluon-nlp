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
"""Text8 dataset."""

__all__ = ['Text8']

import os
import zipfile

from mxnet.gluon.utils import check_sha1, download

from ..dataset import CorpusDataset
from ..utils import _get_home_dir


class Text8(CorpusDataset):
    """Text8 corpus

    http://mattmahoney.net/dc/textdata.html

    Part of the test data for the Large Text Compression Benchmark
    http://mattmahoney.net/dc/text.html. The first 10**8 bytes of the English
    Wikipedia dump on Mar. 3, 2006.

    License: https://en.wikipedia.org/wiki/Wikipedia:Copyrights

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/text8'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    """

    archive_file = ('text8.zip', '6c70299b93b7e1f927b42cd8f6ac1a31547c7a2e')
    data_file = {
        'train': ('text8', '0dc3edebc970dcc96137e7deda4d9995af9d93de')
    }
    url = 'http://mattmahoney.net/dc/'

    def __init__(self, root=os.path.join(_get_home_dir(), 'datasets', 'text8'),
                 segment='train', max_sentence_length=10000):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._max_sentence_length = max_sentence_length
        super(Text8, self).__init__(self._get_data())

        # pylint: disable=access-member-before-definition
        if max_sentence_length:
            data = []
            for sentence in self._data:
                for i in range(0, len(sentence), max_sentence_length):
                    data.append(sentence[i:i + max_sentence_length])
            self._data = data

    def _get_data(self):
        archive_file_name, archive_hash = self.archive_file
        data_file_name, data_hash = self.data_file[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            downloaded_file_path = download(self.url + archive_file_name,
                                            path=root, sha1_hash=archive_hash)

            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                zf.extractall(root)
        return path
