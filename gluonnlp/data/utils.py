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

"""Utility classes and functions. They help organize and keep statistics of datasets."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['Counter', 'count_tokens', 'concat_sequence', 'slice_sequence', 'train_valid_split']

import os
import collections
import zipfile
import numpy as np

from mxnet.gluon.data import SimpleDataset
from mxnet.gluon.utils import _get_repo_url, download, check_sha1

from .. import _constants as C


class Counter(collections.Counter): # pylint: disable=abstract-method
    """Counter class for keeping token frequencies."""

class DefaultLookupDict(dict):
    """Dictionary class with fall-back look-up with default value set in the constructor."""

    def __init__(self, default, d=None):
        if d:
            super(DefaultLookupDict, self).__init__(d)
        else:
            super(DefaultLookupDict, self).__init__()
        self._default = default

    def __getitem__(self, k):
        return self.get(k, self._default)


def count_tokens(tokens, to_lower=False, counter=None):
    r"""Counts tokens in the specified string.

    For token_delim='(td)' and seq_delim='(sd)', a specified string of two sequences of tokens may
    look like::

        (td)token1(td)token2(td)token3(td)(sd)(td)token4(td)token5(td)(sd)


    Parameters
    ----------
    tokens : list of str
        A source list of tokens.
    to_lower : bool, default False
        Whether to convert the source source_str to the lower case.
    counter : Counter or None, default None
        The Counter instance to be updated with the counts of `tokens`. If
        None, return a new Counter instance counting tokens from `tokens`.

    Returns
    -------
    The `counter` Counter instance after being updated with the token
    counts of `source_str`. If `counter` is None, return a new Counter
    instance counting tokens from `source_str`.

    Examples
    --------
    >>> source_str = ' Life is great ! \\n life is good . \\n'
    >>> source_str_tokens = filter(None, re.split(' |\n', source_str))
    >>> count_tokens(source_str_tokens)
    Counter({'!': 1, '.': 1, 'good': 1, 'great': 1, 'is': 2, 'life': 2})

    """
    if to_lower:
        tokens = [t.lower() for t in tokens]

    if counter is None:
        return Counter(tokens)
    else:
        counter.update(tokens)
        return counter


def concat_sequence(sequences):
    """Concatenate sequences of tokens into a single flattened list of tokens.

    Parameters
    ----------
    sequences : list of list of object
        Sequences of tokens, each of which is an iterable of tokens.

    Returns
    -------
    Flattened list of tokens.

    """
    return [token for seq in sequences for token in seq if token]


def slice_sequence(sequence, length, pad_last=False, pad_val=C.PAD_TOKEN, overlap=0):
    """Slice a flat sequence of tokens into sequences tokens, with each
    inner sequence's length equal to the specified `length`, taking into account the requested
    sequence overlap.

    Parameters
    ----------
    sequence : list of object
        A flat list of tokens.
    length : int
        The length of each of the samples.
    pad_last : bool, default False
        Whether to pad the last sequence when its length doesn't align. If the last sequence's
        length doesn't align and ``pad_last`` is False, it will be dropped.
    pad_val : object, default
        The padding value to use when the padding of the last sequence is enabled. In general,
        the type of ``pad_val`` should be the same as the tokens.
    overlap : int, default 0
        The extra number of items in current sample that should overlap with the
        next sample.

    Returns
    -------
    List of list of tokens, with the length of each inner list equal to `length`.

    """
    if length <= overlap:
        raise ValueError('length needs to be larger than overlap')

    if pad_last:
        pad_len = _slice_pad_length(len(sequence), length, overlap)
        sequence = sequence + [pad_val] * pad_len
    num_samples = (len(sequence)-length) // (length-overlap) + 1
    return [sequence[i*(length-overlap):((i+1)*length-i*overlap)] for i in range(num_samples)]


def _slice_pad_length(num_items, length, overlap=0):
    """Calculate the padding length needed for sliced samples in order not to discard data.

    Parameters
    ----------
    num_items : int
        Number of items in dataset before collating.
    length : int
        The length of each of the samples.
    overlap : int, default 0
        The extra number of items in current sample that should overlap with the
        next sample.

    Returns
    -------
    Length of paddings.

    """
    if length <= overlap:
        raise ValueError('length needs to be larger than overlap')

    step = length-overlap
    span = num_items-length
    residual = span % step
    if residual:
        return step - residual
    else:
        return 0


_vocab_sha1 = {'wikitext-2': 'be36dc5238c2e7d69720881647ab72eb506d0131'}
_url_format = '{repo_url}gluon/dataset/vocab/{file_name}.zip'


def train_valid_split(dataset, valid_ratio=0.05):
    """Split the dataset into training and validation sets.

    Parameters
    ----------
    train : list
        A list of training samples.
    valid_ratio : float, default 0.05
        Proportion of training samples to use for validation set
        range: [0, 1]

    Returns
    -------
    train : SimpleDataset
    valid : SimpleDataset
    """
    if not 0.0 <= valid_ratio <= 1.0:
        raise ValueError('valid_ratio should be in [0, 1]')

    num_train = len(dataset)
    num_valid = np.ceil(num_train * valid_ratio).astype('int')
    indices = np.arange(num_train)

    np.random.shuffle(indices)
    valid = SimpleDataset([dataset[indices[i]] for i in range(num_valid)])
    train = SimpleDataset([dataset[indices[i + num_valid]] for i in range(num_train - num_valid)])
    return train, valid


def short_hash(name):
    if name not in _vocab_sha1:
        raise ValueError('Vocabulary for {name} is not available.'.format(name=name))
    return _vocab_sha1[name][:8]

def _load_pretrained_vocab(name, root=os.path.join('~', '.mxnet', 'models')):
    """Load the accompanying vocabulary object for pretrained model.

    Parameters
    ----------
    name : str
        Name of the vocabulary, usually the name of the dataset.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested vocabulary object file.
    """
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.vocab')
    sha1_hash = _vocab_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return _load_vocab_file(file_path)
        else:
            print('Detected mismatch in the content of model vocab file. Downloading again.')
    else:
        print('Vocab file is not found. Downloading.')

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name+'.zip')
    repo_url = _get_repo_url()
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return _load_vocab_file(file_path)
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

def _load_vocab_file(file_path):
    with open(file_path, 'r') as f:
        from ..vocab import Vocab
        return Vocab.from_json(f.read())
