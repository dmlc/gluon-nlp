
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
"""Sentiment analysis datasets."""

__all__ = ['IMDB', 'MR', 'TREC', 'SUBJ', 'SST_1', 'SST_2']

import json
import os

from mxnet.gluon.data import SimpleDataset
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from .registry import register
from ..base import get_home_dir


class SentimentDataset(SimpleDataset):
    """Base class for sentiment analysis data sets.

    Parameters
    ----------
    segment : str
        Dataset segment.
    root : str
        Path to temp folder for storing data.
    """
    def __init__(self, segment, root):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(SentimentDataset, self).__init__(self._read_data())

    def _get_data(self):
        data_file_name, data_hash = self._data_file()[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            download(_get_repo_file_url(self._repo_dir(), data_file_name),
                     path=root, sha1_hash=data_hash)

    def _read_data(self):
        with open(os.path.join(self._root, self._segment + '.json')) as f:
            samples = json.load(f)
        return samples

    def _data_file(self):
        raise NotImplementedError

    def _repo_dir(self):
        raise NotImplementedError


@register(segment=['train', 'test', 'unsup'])
class IMDB(SentimentDataset):
    """IMDB reviews for sentiment analysis.

    From
    http://ai.stanford.edu/~amaas/data/sentiment/

    Positive classes have label values in [7, 10]. Negative classes have label values in [1, 4].
    All samples in unsupervised set have labels with value 0.

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'test', and 'unsup' for unsupervised.
    root : str, default '$MXNET_HOME/datasets/imdb'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> imdb = gluonnlp.data.IMDB('test', root='./datasets/imdb')
    -etc-
    >>> len(imdb)
    25000
    >>> len(imdb[0])
    2
    >>> type(imdb[0][0]), type(imdb[0][1])
    (<class 'str'>, <class 'int'>)
    >>> imdb[0][0][:75]
    'I went and saw this movie last night after being coaxed to by a few friends'
    >>> imdb[0][1]
    10
    >>> imdb = gluonnlp.data.IMDB('unsup', root='./datasets/imdb')
    -etc-
    >>> len(imdb)
    50000
    >>> len(imdb[0])
    2
    >>> type(imdb[0][0]), type(imdb[0][1])
    (<class 'str'>, <class 'int'>)
    >>> imdb[0][0][:70]
    'I admit, the great majority of films released before say 1933 are just'
    >>> imdb[0][1]
    0
    """
    def __init__(self, segment='train', root=os.path.join(get_home_dir(), 'datasets', 'imdb')):
        super(IMDB, self).__init__(segment, root)

    def _data_file(self):
        return {'train': ('train.json', '516a0ba06bca4e32ee11da2e129f4f871dff85dc'),
                'test': ('test.json', '7d59bd8899841afdc1c75242815260467495b64a'),
                'unsup': ('unsup.json', 'f908a632b7e7d7ecf113f74c968ef03fadfc3c6c')}

    def _repo_dir(self):
        return 'gluon/dataset/imdb'


@register()
class MR(SentimentDataset):
    """Movie reviews for sentiment analysis.

    From
    https://www.cs.cornell.edu/people/pabo/movie-review-data/

    Positive class has label value 1. Negative class has label value 0.

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/mr'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> mr = gluonnlp.data.MR(root='./datasets/mr')
    -etc-
    >>> len(mr)
    10662
    >>> len(mr[3])
    2
    >>> type(mr[3][0]), type(mr[3][1])
    (<class 'str'>, <class 'int'>)
    >>> mr[3][0][:55]
    'if you sometimes like to go to the movies to have fun ,'
    >>> mr[3][1]
    1
    """
    def __init__(self, root=os.path.join(get_home_dir(), 'datasets', 'mr')):
        super(MR, self).__init__('all', root)

    def _data_file(self):
        return {'all': ('all.json', '7606efec578d9613f5c38bf2cef8d3e4e6575b2c')}

    def _repo_dir(self):
        return 'gluon/dataset/mr'


@register(segment=['train', 'test'])
class TREC(SentimentDataset):
    """Question dataset for question classification.

    From
    http://cogcomp.cs.illinois.edu/Data/QA/QC/

    Class labels are (http://cogcomp.org/Data/QA/QC/definition.html):
        - DESCRIPTION: 0
        - ENTITY: 1
        - ABBREVIATION: 2
        - HUMAN: 3
        - LOCATION: 4
        - NUMERIC: 5

    The first space-separated token in the text of each sample is the fine-grain label.

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'test'.
    root : str, default '$MXNET_HOME/datasets/trec'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> trec = gluonnlp.data.TREC('test', root='./datasets/trec')
    -etc-
    >>> len(trec)
    500
    >>> len(trec[0])
    2
    >>> type(trec[0][0]), type(trec[0][1])
    (<class 'str'>, <class 'int'>)
    >>> trec[0][0]
    'ind Who was the first US President to ride in an automobile to his inauguration ?'
    >>> (trec[0][1], trec[0][0].split()[0])
    (3, 'ind')
    """
    def __init__(self, segment='train', root=os.path.join(get_home_dir(), 'datasets', 'trec')):
        super(TREC, self).__init__(segment, root)

    def _data_file(self):
        return {'train': ('train.json', 'f764e8e052239c66e96e15133c8fc4028df34a84'),
                'test': ('test.json', 'df8c6ffb90831e553617dbaab7119e0526b98f35')}

    def _repo_dir(self):
        return 'gluon/dataset/trec'


@register()
class SUBJ(SentimentDataset):
    """Subjectivity dataset for sentiment analysis.

    Positive class has label value 1. Negative class has label value 0.

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/subj'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> subj = gluonnlp.data.SUBJ(root='./datasets/subj')
    -etc-
    >>> len(subj)
    10000
    >>> len(subj[0])
    2
    >>> type(subj[0][0]), type(subj[0][1])
    (<class 'str'>, <class 'int'>)
    >>> subj[0][0][:60]
    'its impressive images of crematorium chimney fires and stack'
    >>> subj[0][1]
    1
    """
    def __init__(self, root=os.path.join(get_home_dir(), 'datasets', 'subj')):
        super(SUBJ, self).__init__('all', root)

    def _data_file(self):
        return {'all': ('all.json', '9e7bd1daa359c24abe1fac767d0e0af7bc114045')}

    def _repo_dir(self):
        return 'gluon/dataset/subj'


@register(segment=['train', 'test'])
class SST_1(SentimentDataset):
    """Stanford Sentiment Treebank: an extension of the MR data set.
    However, train/dev/test splits are provided and labels are fine-grained
    (very positive, positive, neutral, negative, very negative).

    From
    http://nlp.stanford.edu/sentiment/

    Class labels are:
        - very positive: 4
        - positive: 3
        - neutral: 2
        - negative: 1
        - very negative: 0

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'test'.
    root : str, default '$MXNET_HOME/datasets/sst-1'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> sst_1 = gluonnlp.data.SST_1('test', root='./datasets/sst_1')
    -etc-
    >>> len(sst_1)
    2125
    >>> len(sst_1[0])
    2
    >>> type(sst_1[0][0]), type(sst_1[0][1])
    (<class 'str'>, <class 'int'>)
    >>> sst_1[0][0][:73]
    "I 've heard that the fans of the first Men in Black have come away hating"
    >>> sst_1[0][1]
    2
    """
    def __init__(self, segment='train', root=os.path.join(get_home_dir(), 'datasets', 'sst-1')):
        super(SST_1, self).__init__(segment, root)

    def _data_file(self):
        return {'train': ('train.json', 'c369d7b1e46134e87e18eb5a1cadf0f2bfcd1787'),
                'test': ('test.json', 'a6999ca5f3d51b61f63ee2ede03ff72e699ac20e')}

    def _repo_dir(self):
        return 'gluon/dataset/sst-1'


@register(segment=['train', 'test'])
class SST_2(SentimentDataset):
    """Stanford Sentiment Treebank: an extension of the MR data set.
    Same as the SST-1 data set except that neutral reviews are removed
    and labels are binary (positive, negative).

    From
    http://nlp.stanford.edu/sentiment/

    Positive class has label value 1. Negative class has label value 0.

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'test'.
    root : str, default '$MXNET_HOME/datasets/sst-2'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> sst_2 = gluonnlp.data.SST_2('test', root='./datasets/sst_2')
    -etc-
    >>> len(sst_2)
    1745
    >>> len(sst_2[0])
    2
    >>> type(sst_2[0][0]), type(sst_2[0][1])
    (<class 'str'>, <class 'int'>)
    >>> sst_2[0][0][:65]
    "Here 's a British flick gleefully unconcerned with plausibility ,"
    >>> sst_2[0][1]
    1
    """
    def __init__(self, segment='train', root=os.path.join(get_home_dir(), 'datasets', 'sst-2')):
        super(SST_2, self).__init__(segment, root)

    def _data_file(self):
        return {'train': ('train.json', '12f4fb2661ad8e39daa45a3369bedb0cd49ad1f4'),
                'test': ('test.json', '34dfb27ef788599a0c424d05a97c1c4389f68c85')}

    def _repo_dir(self):
        return 'gluon/dataset/sst-2'
