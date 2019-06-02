
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

__all__ = ['IMDB', 'MR', 'TREC', 'SUBJ', 'SST_1', 'SST_2', 'CR', 'MPQA']

import json
import os
import shutil
import zipfile

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
        """Load data from the file. Do nothing if data was loaded before.
        """
        (data_archive_name, archive_hash), (data_name, data_hash) \
            = self._data_file()[self._segment]
        data_path = os.path.join(self._root, data_name)

        if not os.path.exists(data_path) or not check_sha1(data_path, data_hash):
            file_path = download(_get_repo_file_url(self._repo_dir(), data_archive_name),
                                 path=self._root, sha1_hash=archive_hash)

            with zipfile.ZipFile(file_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    if filename:
                        dest = os.path.join(self._root, filename)
                        with zf.open(member) as source, open(dest, 'wb') as target:
                            shutil.copyfileobj(source, target)

    def _read_data(self):
        (_, _), (data_file_name, _) = self._data_file()[self._segment]

        with open(os.path.join(self._root, data_file_name)) as f:
            samples = json.load(f)
        return samples

    def _data_file(self):
        raise NotImplementedError

    def _repo_dir(self):
        raise NotImplementedError


@register(segment=['train', 'test', 'unsup'])
class IMDB(SimpleDataset):
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
        self._data_file = {'train': ('train.json',
                                     '516a0ba06bca4e32ee11da2e129f4f871dff85dc'),
                           'test': ('test.json',
                                    '7d59bd8899841afdc1c75242815260467495b64a'),
                           'unsup': ('unsup.json',
                                     'f908a632b7e7d7ecf113f74c968ef03fadfc3c6c')}
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(IMDB, self).__init__(self._read_data())

    def _get_data(self):
        data_file_name, data_hash = self._data_file[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            download(_get_repo_file_url('gluon/dataset/imdb', data_file_name),
                     path=root, sha1_hash=data_hash)

    def _read_data(self):
        with open(os.path.join(self._root, self._segment+'.json')) as f:
            samples = json.load(f)
        return samples


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
        return {'all': (('all-7606efec.zip', '0fcbaffe0bac94733e6497f700196585f03fa89e'),
                        ('all-7606efec.json', '7606efec578d9613f5c38bf2cef8d3e4e6575b2c '))}

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
    'How far is it from Denver to Aspen ?'
    >>> (trec[0][1], trec[0][0].split()[0])
    (5, 'How')
    """
    def __init__(self, segment='train', root=os.path.join(get_home_dir(), 'datasets', 'trec')):
        super(TREC, self).__init__(segment, root)

    def _data_file(self):
        return {'train': (('train-1776132f.zip', '337d3f43a56ec26f5773c6fc406ef19fb4cd3c92'),
                          ('train-1776132f.json', '1776132fb2fc0ed2dc91b62f7817a4e071a3c7de')),
                'test': (('test-ff9ad0ce.zip', '57f03aaee2651ca05f1f9fc5731ba7e9ad98e38a'),
                         ('test-ff9ad0ce.json', 'ff9ad0ceb44d8904663fee561804a8dd0edc1b15'))}

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
        return {'all': (('all-9e7bd1da.zip', '8b0d95c2fc885cc38e4ad776d7429183f3ef632b'),
                        ('all-9e7bd1da.json', '9e7bd1daa359c24abe1fac767d0e0af7bc114045'))}

    def _repo_dir(self):
        return 'gluon/dataset/subj'


@register(segment=['train', 'dev', 'test'])
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
    2210
    >>> len(sst_1[0])
    2
    >>> type(sst_1[0][0]), type(sst_1[0][1])
    (<class 'str'>, <class 'int'>)
    >>> sst_1[0][0][:73]
    'no movement , no yuks , not much of anything .'
    >>> sst_1[0][1]
    1
    """
    def __init__(self, segment='train', root=os.path.join(get_home_dir(), 'datasets', 'sst-1')):
        super(SST_1, self).__init__(segment, root)

    def _data_file(self):
        return {'train': (('train-638f9352.zip', '0a039010449772700c0e270c7095362403dc486a'),
                          ('train-638f9352.json', '638f935251c0474e93d4aa50fda0c900faf02bba')),
                'dev': (('dev-820ac954.zip', 'e4b7899ef5d37a6bf01d8ec1115ba20b8419b96f'),
                        ('dev-820ac954.json', '820ac954b14b4f7d947e25f7a99249618d7962ee')),
                'test': (('test-ab593ae9.zip', 'd3736db56cdc7293c38435557697c2407652525d'),
                         ('test-ab593ae9.json', 'ab593ae9628f94af4f698654158ded1488b1de3b'))}

    def _repo_dir(self):
        return 'gluon/dataset/sst-1'


@register(segment=['train', 'dev', 'test'])
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
    1821
    >>> len(sst_2[0])
    2
    >>> type(sst_2[0][0]), type(sst_2[0][1])
    (<class 'str'>, <class 'int'>)
    >>> sst_2[0][0][:65]
    'no movement , no yuks , not much of anything .'
    >>> sst_2[0][1]
    0
    """
    def __init__(self, segment='train', root=os.path.join(get_home_dir(), 'datasets', 'sst-2')):
        super(SST_2, self).__init__(segment, root)

    def _data_file(self):
        return {'train': (('train-61f1f238.zip', 'f27a9ac6a7c9208fb7f024b45554da95639786b3'),
                          ('train-61f1f238.json', '61f1f23888652e11fb683ac548ed0be8a87dddb1')),
                'dev': (('dev-65511587.zip', '8c74911f0246bd88dc0ced2619f95f10db09dc98'),
                        ('dev-65511587.json', '655115875d83387b61f9701498143724147a1fc9')),
                'test': (('test-a39c1db6.zip', '4b7f1648207ec5dffb4e4783cf1f48d6f36ba4db'),
                         ('test-a39c1db6.json', 'a39c1db6ecc3be20bf2563bf2440c3c06887a2df'))}

    def _repo_dir(self):
        return 'gluon/dataset/sst-2'

@register()
class CR(SentimentDataset):
    """
    Customer reviews of various products (cameras, MP3s etc.). The task is to
    predict positive/negative reviews.

    Positive class has label value 1. Negative class has label value 0.

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/cr'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> cr = gluonnlp.data.CR(root='./datasets/cr')
    -etc-
    >>> len(cr)
    3775
    >>> len(cr[3])
    2
    >>> type(cr[3][0]), type(cr[3][1])
    (<class 'str'>, <class 'int'>)
    >>> cr[3][0][:55]
    'i know the saying is " you get what you pay for " but a'
    >>> cr[3][1]
    0
    """
    def __init__(self, root=os.path.join(get_home_dir(), 'datasets', 'cr')):
        super(CR, self).__init__('all', root)

    def _data_file(self):
        return {'all': (('all-0c9633c6.zip', 'c662e2f9115d74e1fcc7c896fa3e2dc5ee7688e7'),
                        ('all-0c9633c6.json', '0c9633c695d29b18730eddff965c850425996edf'))}

    def _repo_dir(self):
        return 'gluon/dataset/cr'

@register()
class MPQA(SentimentDataset):
    """
    Opinion polarity detection subtask of the MPQA dataset.

    From
    http://www.cs.pitt.edu/mpqa/

    Positive class has label value 1. Negative class has label value 0.

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/mpqa'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> mpqa = gluonnlp.data.MPQA(root='./datasets/mpqa')
    -etc-
    >>> len(mpqa)
    10606
    >>> len(mpqa[3])
    2
    >>> type(mpqa[3][0]), type(mpqa[3][1])
    (<class 'str'>, <class 'int'>)
    >>> mpqa[3][0][:25]
    'many years of decay'
    >>> mpqa[3][1]
    0
    """
    def __init__(self, root=os.path.join(get_home_dir(), 'datasets', 'mpqa')):
        super(MPQA, self).__init__('all', root)

    def _data_file(self):
        return {'all': (('all-bcbfeed8.zip', 'e07ae226cfe4713328eeb9660b261b9852ff5865'),
                        ('all-bcbfeed8.json', 'bcbfeed8b8767a564bdc428486ef18c1ba4dc536'))}

    def _repo_dir(self):
        return 'gluon/dataset/mpqa'
