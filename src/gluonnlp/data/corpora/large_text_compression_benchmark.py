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
"""Large Text Compression Benchmark.

The test data for the Large Text Compression Benchmark is the first 109 bytes
of the English Wikipedia dump on Mar. 3, 2006.
http://download.wikipedia.org/enwiki/20060303/enwiki-20060303-pages-articles.xml.bz2
(1.1 GB or 4.8 GB after decompressing with bzip2 - link no longer works).
Results are also given for the first 108 bytes, which is also used for the
Hutter Prize. These files have the following sizes and checksums:

File     Size (bytes)   MD5 (GNU md5sum 1.22)             SHA-1 (SlavaSoft fsum 2.51)
------   -------------  --------------------------------  ----------------------------------------
enwik8     100,000,000  a1fa5ffddb56f4953e226637dabbb36a  57b8363b814821dc9d47aa4d41f58733519076b2
enwik9   1,000,000,000  e206c3450ac99950df65bf70ef61a12d  2996e86fb978f93cca8f566cc56998923e7fe581

See http://mattmahoney.net/dc/text.html and
http://mattmahoney.net/dc/textdata.html for more information.

"""

__all__ = ['Text8', 'Fil9']

import os
import zipfile

from mxnet.gluon.utils import _get_repo_file_url, check_sha1, download

from ..dataset import CorpusDataset
from ..utils import _get_home_dir


class _LargeTextCompressionBenchmark(CorpusDataset):
    def __init__(self, root, segment, **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._namespace = 'gluon/dataset/large_text_compression_benchmark'
        super(_LargeTextCompressionBenchmark, self).__init__(
            self._get_data(), **kwargs)

    def _get_data(self):
        archive_file_name, archive_hash = self.archive_file
        data_file_name, data_hash = self.data_file[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            downloaded_file_path = download(
                _get_repo_file_url(self._namespace, archive_file_name),
                path=root,
                sha1_hash=archive_hash)

            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                zf.extractall(root)
        return path


class Text8(_LargeTextCompressionBenchmark):
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

    archive_file = ('text8-6c70299b.zip', '6c70299b93b7e1f927b42cd8f6ac1a31547c7a2e')
    data_file = {
        'train': ('text8', '0dc3edebc970dcc96137e7deda4d9995af9d93de')
    }

    def __init__(self,
                 root=os.path.join(_get_home_dir(), 'datasets', 'text8'),
                 segment='train',
                 max_sentence_length=10000):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._max_sentence_length = max_sentence_length
        super(Text8, self).__init__(root=root, segment=segment)

        # pylint: disable=access-member-before-definition
        if max_sentence_length:
            data = []
            for sentence in self._data:
                for i in range(0, len(sentence), max_sentence_length):
                    data.append(sentence[i:i + max_sentence_length])
            self._data = data


class Fil9(_LargeTextCompressionBenchmark):
    """Fil9 corpus

    http://mattmahoney.net/dc/textdata.html

    Part of the test data for the Large Text Compression Benchmark
    http://mattmahoney.net/dc/text.html. The first 10**9 bytes of the English
    Wikipedia dump on Mar. 3, 2006.

    License: https://en.wikipedia.org/wiki/Wikipedia:Copyrights

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/fil9'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    """

    archive_file = ('fil9-e2a6a602.zip',
                    'e2a6a602be8d3f9712c92423581aa47e7ffd5906')
    data_file = {'train': ('fil9', '08caf9b1d5600233aa19cb6b25d7b798558304d3')}

    def __init__(self,
                 root=os.path.join(_get_home_dir(), 'datasets', 'fil9'),
                 segment='train',
                 max_sentence_length=None):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._max_sentence_length = max_sentence_length
        super(Fil9, self).__init__(root=root, segment=segment)

        # pylint: disable=access-member-before-definition
        if max_sentence_length is not None:
            data = []
            for sentence in self._data:
                for i in range(0, len(sentence), max_sentence_length):
                    data.append(sentence[i:i + max_sentence_length])
            self._data = data
