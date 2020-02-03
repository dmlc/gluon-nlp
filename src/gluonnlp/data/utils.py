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
import collections
import os
import tarfile
import zipfile
import random
import sys
import shutil

import numpy as np
from mxnet.gluon.data import SimpleDataset
from mxnet.gluon.utils import _get_repo_url, check_sha1, download

from .. import _constants as C
from .. import utils

__all__ = [
    'Counter', 'count_tokens', 'concat_sequence', 'slice_sequence', 'train_valid_split',
    'line_splitter', 'whitespace_splitter', 'Splitter'
]


class Counter(collections.Counter):  # pylint: disable=abstract-method
    """Counter class for keeping token frequencies."""

    def discard(self, min_freq, unknown_token):
        """Discards tokens with frequency below min_frequency and represents them
        as `unknown_token`.

        Parameters
        ----------
        min_freq: int
            Tokens whose frequency is under min_freq is counted as `unknown_token` in
            the Counter returned.
        unknown_token: str
            The representation for any unknown token.

        Returns
        -------
        The Counter instance.

        Examples
        --------
        >>> a = gluonnlp.data.Counter({'a': 10, 'b': 1, 'c': 1})
        >>> a.discard(3, '<unk>')
        Counter({'a': 10, '<unk>': 2})
        """
        freq = 0
        ret = Counter({})
        for token, count in self.items():
            if count < min_freq:
                freq += count
            else:
                ret[token] = count
        ret[unknown_token] = ret.get(unknown_token, 0) + freq
        return ret


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
    >>> import re
    >>> source_str = ' Life is great ! \n life is good . \n'
    >>> source_str_tokens = filter(None, re.split(' |\n', source_str))
    >>> counter = gluonnlp.data.count_tokens(source_str_tokens)
    >>> sorted(counter.items())
    [('!', 1), ('.', 1), ('Life', 1), ('good', 1), ('great', 1), ('is', 2), ('life', 1)]

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
    num_samples = (len(sequence) - length) // (length - overlap) + 1

    return [sequence[i * (length - overlap): ((i + 1) * length - i * overlap)]
            for i in range(num_samples)]


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

    step = length - overlap
    span = num_items - length
    residual = span % step
    if residual:
        return step - residual
    else:
        return 0


# name:[sha hash, file extension, special tokens]
_vocab_sha1 = {'wikitext-2':
               ['be36dc5238c2e7d69720881647ab72eb506d0131', '.vocab', {}],
               'gbw':
               ['ebb1a287ca14d8fa6f167c3a779e5e7ed63ac69f', '.vocab', {}],
               'WMT2014_src':
               ['230ebb817b1d86950d71e2e765f192a4e4f34415', '.vocab', {}],
               'WMT2014_tgt':
               ['230ebb817b1d86950d71e2e765f192a4e4f34415', '.vocab', {}],
               'book_corpus_wiki_en_cased':
               ['2d62af22535ed51f35cc8e2abb607723c89c2636', '.vocab', {}],
               'book_corpus_wiki_en_uncased':
               ['a66073971aa0b1a262453fe51342e57166a8abcf', '.vocab', {}],
               'openwebtext_book_corpus_wiki_en_uncased':
               ['a66073971aa0b1a262453fe51342e57166a8abcf', '.vocab', {}],
               'openwebtext_ccnews_stories_books_cased':
               ['2b804f8f90f9f93c07994b703ce508725061cf43', '.vocab', {}],
               'wiki_multilingual_cased':
               ['0247cb442074237c38c62021f36b7a4dbd2e55f7', '.vocab', {}],
               'distilbert_book_corpus_wiki_en_uncased':
               ['80ef760a6bdafec68c99b691c94ebbb918c90d02', '.vocab', {}],
               'wiki_cn_cased':
               ['ddebd8f3867bca5a61023f73326fb125cf12b4f5', '.vocab', {}],
               'wiki_multilingual_uncased':
               ['2b2514cc539047b9179e9d98a4e68c36db05c97a', '.vocab', {}],
               'scibert_scivocab_uncased':
               ['2d2566bfc416790ab2646ab0ada36ba628628d60', '.vocab', {}],
               'scibert_scivocab_cased':
               ['2c714475b521ab8542cb65e46259f6bfeed8041b', '.vocab', {}],
               'scibert_basevocab_uncased':
               ['80ef760a6bdafec68c99b691c94ebbb918c90d02', '.vocab', {}],
               'scibert_basevocab_cased':
               ['a4ff6fe1f85ba95f3010742b9abc3a818976bb2c', '.vocab', {}],
               'biobert_v1.0_pmc_cased':
               ['a4ff6fe1f85ba95f3010742b9abc3a818976bb2c', '.vocab', {}],
               'biobert_v1.0_pubmed_cased':
               ['a4ff6fe1f85ba95f3010742b9abc3a818976bb2c', '.vocab', {}],
               'biobert_v1.0_pubmed_pmc_cased':
               ['a4ff6fe1f85ba95f3010742b9abc3a818976bb2c', '.vocab', {}],
               'biobert_v1.1_pubmed_cased':
               ['a4ff6fe1f85ba95f3010742b9abc3a818976bb2c', '.vocab', {}],
               'clinicalbert_uncased':
               ['80ef760a6bdafec68c99b691c94ebbb918c90d02', '.vocab', {}],
               'baidu_ernie_uncased':
               ['223553643220255e2a0d4c60e946f4ad7c719080', '.vocab', {}],
               'openai_webtext':
               ['f917dc7887ce996068b0a248c8d89a7ec27b95a1', '.vocab', {}],
               'xlnet_126gb':
               ['0d74490383bbc5c62b8bcea74d8b74a1bb1280b3', '.vocab', {}],
               'kobert_news_wiki_ko_cased':
               ['f86b1a8355819ba5ab55e7ea4a4ec30fdb5b084f', '.spiece', {'padding_token': '[PAD]'}]}

_url_format = '{repo_url}gluon/dataset/vocab/{file_name}.zip'


def train_valid_split(dataset, valid_ratio=0.05, stratify=None):
    """Split the dataset into training and validation sets.

    Parameters
    ----------
    dataset : list
        A list of training samples.
    valid_ratio : float, default 0.05
        Proportion of training samples to use for validation set
        range: [0, 1]
    stratify : list, default None
        If not None, data is split in a stratified fashion,
        using the contents of stratify as class labels.

    Returns
    -------
    train : SimpleDataset
    valid : SimpleDataset
    """
    if not 0.0 <= valid_ratio <= 1.0:
        raise ValueError('valid_ratio should be in [0, 1]')

    if not stratify:
        num_train = len(dataset)
        num_valid = np.ceil(num_train * valid_ratio).astype('int')
        indices = np.arange(num_train)

        np.random.shuffle(indices)
        valid = SimpleDataset([dataset[indices[i]] for i in range(num_valid)])
        train = SimpleDataset(
            [dataset[indices[i + num_valid]] for i in range(num_train - num_valid)])

        return train, valid
    else:
        if not isinstance(stratify, list):
            raise TypeError('stratify should be a list')
        if not len(stratify) == len(dataset):
            raise ValueError('stratify should be the same length as num_train')

        classes, digitized = np.unique(stratify, return_inverse=True)
        n_classes = len(classes)
        num_class = np.bincount(digitized)
        num_valid = np.ceil(valid_ratio * num_class).astype('int')

        valid = []
        train = []

        for idx in range(n_classes):
            indices = np.nonzero(stratify == classes[idx])[0]
            np.random.shuffle(indices)
            valid += [dataset[indices[i]] for i in range(num_valid[idx])]
            train += [dataset[indices[i + num_valid[idx]]]
                      for i in range(num_class[idx] - num_valid[idx])]

        np.random.shuffle(valid)
        np.random.shuffle(train)

        train = SimpleDataset(train)
        valid = SimpleDataset(valid)

        return train, valid


def short_hash(name):
    if name not in _vocab_sha1:
        vocabs = list(_vocab_sha1.keys())
        raise ValueError('Vocabulary for {name} is not available. '
                         'Hosted vocabularies include: {vocabs}'.format(name=name,
                                                                        vocabs=vocabs))
    return _vocab_sha1[name][0][:8]


def _get_vocab_tokenizer_info(name, root):
    file_name = '{name}-{short_hash}'.format(name=name,
                                             short_hash=short_hash(name))
    root = os.path.expanduser(root)
    sha1_hash, file_ext, special_tokens = _vocab_sha1[name]
    return file_name, file_ext, sha1_hash, special_tokens


def _download_vocab_tokenizer(root, file_name, file_ext, file_path):
    utils.mkdir(root)

    temp_num = str(random.Random().randint(1, sys.maxsize))
    temp_root = os.path.join(root, temp_num)
    temp_file_path = os.path.join(temp_root, file_name + file_ext)
    temp_zip_file_path = os.path.join(temp_root, temp_num + '_' + file_name + '.zip')

    repo_url = _get_repo_url()
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=temp_zip_file_path, overwrite=True)
    with zipfile.ZipFile(temp_zip_file_path) as zf:
        assert file_name + file_ext in zf.namelist(), '{} not part of {}. Only have: {}'.format(
            file_name + file_ext, file_name + '.zip', zf.namelist())
        utils.mkdir(temp_root)
        zf.extractall(temp_root)
        os.replace(temp_file_path, file_path)
        shutil.rmtree(temp_root)

def _load_pretrained_vocab(name, root, cls=None):
    """Load the accompanying vocabulary object for pre-trained model.

    Parameters
    ----------
    name : str
        Name of the vocabulary, usually the name of the dataset.
    root : str
        Location for keeping the model vocabulary.
    cls : nlp.Vocab or nlp.vocab.BERTVocab, default nlp.Vocab

    Returns
    -------
    Vocab or nlp.vocab.BERTVocab, Tokenizer or None
        Loaded vocabulary object and Tokenizer for the pre-trained model.
    """
    file_name, file_ext, sha1_hash, special_tokens = _get_vocab_tokenizer_info(name, root)
    file_path = os.path.join(root, file_name + file_ext)
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return _load_vocab_file(file_path, cls, **special_tokens)
        else:
            print('Detected mismatch in the content of model vocab file. Downloading again.')
    else:
        print('Vocab file is not found. Downloading.')
    _download_vocab_tokenizer(root, file_name, file_ext, file_path)
    if check_sha1(file_path, sha1_hash):
        return _load_vocab_file(file_path, cls, **special_tokens)
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')


def _load_pretrained_sentencepiece_tokenizer(name, root, **kwargs):
    from ..data import SentencepieceTokenizer  # pylint: disable=import-outside-toplevel
    file_name, file_ext, sha1_hash, _ = _get_vocab_tokenizer_info(name, root)
    file_path = os.path.join(root, file_name + file_ext)
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            assert file_path.endswith('.spiece')
            return SentencepieceTokenizer(file_path, **kwargs)
        else:
            print('Detected mismatch in the content of model tokenizer file. Downloading again.')
    else:
        print('tokenizer file is not found. Downloading.')
    _download_vocab_tokenizer(root, file_name, file_ext, file_path)
    if check_sha1(file_path, sha1_hash):
        assert file_path.endswith('.spiece')
        return SentencepieceTokenizer(file_path, **kwargs)
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')


def _load_vocab_file(file_path, cls, **kwargs):
    with open(file_path, 'r') as f:
        if cls is None:
            from ..vocab import Vocab  # pylint: disable=import-outside-toplevel
            cls = Vocab
        if file_path.endswith('.spiece'):
            assert kwargs is not None, 'special tokens must be specified when .spiece provide.'
            from ..vocab import BERTVocab  # pylint: disable=import-outside-toplevel
            return BERTVocab.from_sentencepiece(
                file_path,
                **kwargs)
        else:
            return cls.from_json(f.read())


def _extract_archive(file, target_dir):  # pylint: disable=redefined-builtin
    """Extract archive file

    Parameters
    ----------
    file : str
        Absolute path of the archive file.
    target_dir : str
        Target directory of the archive to be uncompressed

    """
    if file.endswith('.gz') or file.endswith('.tar') or file.endswith('.tgz'):
        archive = tarfile.open(file, 'r')
    elif file.endswith('.zip'):
        archive = zipfile.ZipFile(file, 'r')
    else:
        raise Exception('Unrecognized file type: ' + file)
    archive.extractall(path=target_dir)
    archive.close()


def line_splitter(s):
    """Split a string at newlines.

    Parameters
    ----------
    s : str
        The string to be split

    Returns
    --------
    List[str]
        List of strings. Obtained by calling s.splitlines().

    """
    return s.splitlines()


def whitespace_splitter(s):
    """Split a string at whitespace (space, tab, newline, return, formfeed).

    Parameters
    ----------
    s : str
        The string to be split

    Returns
    --------
    List[str]
        List of strings. Obtained by calling s.split().
    """
    return s.split()


class Splitter:
    """Split a string based on a separator.

    Parameters
    ----------
    separator : str
        The separator based on which string is split.
    """

    def __init__(self, separator=None):
        self._separator = separator

    def __call__(self, s):
        """Split a string based on the separator.

        Parameters
        ----------
        s : str
            The string to be split

        Returns
        --------
        List[str]
            List of strings. Obtained by calling s.split(separator).
        """
        return s.split(self._separator)
