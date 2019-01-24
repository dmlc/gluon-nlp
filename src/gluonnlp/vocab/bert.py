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

# pylint: disable=consider-iterating-dictionary

"""Vocabulary class used in the BERT."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['BERTVocab']

import json
import warnings

from .vocab import Vocab
from ..data.utils import DefaultLookupDict


class BERTVocab(Vocab):
    """BERT special character vocabulary.

    Parameters
    ----------
    counter : Counter or None, default None
        Counts text token frequencies in the text data. Its keys will be indexed according to
        frequency thresholds such as `max_size` and `min_freq`. Keys of `counter`,
        `unknown_token`, and values of `reserved_tokens` must be of the same hashable type.
        Examples: str, int, and tuple.
    max_size : None or int, default None
        The maximum possible number of the most frequent tokens in the keys of `counter` that can be
        indexed. Note that this argument does not count any token from `reserved_tokens`. Suppose
        that there are different keys of `counter` whose frequency are the same, if indexing all of
        them will exceed this argument value, such keys will be indexed one by one according to
        their __cmp__() order until the frequency threshold is met. If this argument is None or
        larger than its largest possible value restricted by `counter` and `reserved_tokens`, this
        argument has no effect.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to be indexed.
    unknown_token : hashable object or None, default '[unk]'
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation. If None, looking up an unknown token will result in KeyError.
    padding_token : hashable object or None, default '[pad]'
        The representation for the special token of padding token.
    bos_token : hashable object or None, default None
        The representation for the special token of beginning-of-sequence token.
    eos_token : hashable object or None, default None
        The representation for the special token of end-of-sequence token.
    mask_token : hashable object or None, default '[MASK]'
        The representation for the special token of mask token for BERT.
    sep_token : hashable object or None, default '[SEP]'
        A token used to separate sentence pairs for BERT.
    cls_token : hashable object or None, default '[CLS]'
        Classification symbol for BERT.
    reserved_tokens : list of hashable objects or None, default [mask_token, sep_token, cls_token],
        A list of reserved tokens (excluding `unknown_token`) that will always be indexed, such as
        special symbols representing padding, beginning of sentence, and end of sentence. It cannot
        contain `unknown_token` or duplicate reserved tokens. Keys of `counter`, `unknown_token`,
        and values of `reserved_tokens` must be of the same hashable type. Examples: str, int, and
        tuple.

    Attributes
    ----------
    UNKNOWN_TOKEN : '[UNK]'
        Static attribute of BERTVocab, unknown token for BERT.
    PADDING_TOKEN : '[PAD]'
        Static attribute of BERTVocab, padding token for BERT.
    MASK_TOKEN : '[MASK]'
        Static attribute of BERTVocab, mask token for BERT.
    SEP_TOKEN : '[SEP]'
        Static attribute of BERTVocab, a token used to separate sentence pairs for BERT.
    CLS_TOKEN : '[CLS]'
        Static attribute of BERTVocab, Classification symbol for BERT.
    embedding : instance of :class:`gluonnlp.embedding.TokenEmbedding`
        The embedding of the indexed tokens.
    idx_to_token : list of strs
        A list of indexed tokens where the list indices and the token indices are aligned.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    token_to_idx : dict mapping str to int
        A dict mapping each token to its index integer.
    unknown_token : hashable object or None, default '[UNK]'
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    padding_token : hashable object or None, default '[PAD]'
        The representation for padding token.
    bos_token : hashable object or None, default None
        The representation for beginning-of-sentence token.
    eos_token : hashable object or None, default None
        The representation for end-of-sentence token.
    mask_token : hashable object or None, default '[MASK]'
        The representation for the special token of mask token for BERT.
    sep_token : hashable object or None, default '[SEP]'
        a token used to separate sentence pairs for BERT.
    cls_token : hashable object or None, default '[CLS]'
    """

    UNKNOWN_TOKEN = '[UNK]'

    PADDING_TOKEN = '[PAD]'

    MASK_TOKEN = '[MASK]'

    SEP_TOKEN = '[SEP]'

    CLS_TOKEN = '[CLS]'

    def __init__(self, counter=None, max_size=None, min_freq=1, unknown_token=UNKNOWN_TOKEN,
                 padding_token=PADDING_TOKEN, bos_token=None, eos_token=None,
                 mask_token=MASK_TOKEN, sep_token=SEP_TOKEN, cls_token=CLS_TOKEN,
                 reserved_tokens=None):
        special_tokens = [token for token in [mask_token, sep_token, cls_token]
                          if token is not None]
        if reserved_tokens:
            reserved_tokens.extend(special_tokens)
        else:
            reserved_tokens = special_tokens
        super(BERTVocab, self).__init__(counter, max_size, min_freq, unknown_token,
                                        padding_token, bos_token, eos_token, reserved_tokens)
        self._mask_token = mask_token
        self._sep_token = sep_token
        self._cls_token = cls_token

    @property
    def mask_token(self):
        return self._mask_token

    @property
    def sep_token(self):
        return self._sep_token

    @property
    def cls_token(self):
        return self._cls_token

    def to_json(self):
        """Serialize BERTVocab object to json string.
        This method does not serialize the underlying embedding.
        """
        if self._embedding:
            warnings.warn('Serialization of attached embedding '
                          'to json is not supported. '
                          'You may serialize the embedding to a binary format '
                          'separately using bert_vocab.embedding.serialize')
        vocab_dict = {}
        vocab_dict['idx_to_token'] = self._idx_to_token
        vocab_dict['token_to_idx'] = dict(self._token_to_idx)
        vocab_dict['reserved_tokens'] = self._reserved_tokens
        vocab_dict['unknown_token'] = self._unknown_token
        vocab_dict['padding_token'] = self._padding_token
        vocab_dict['bos_token'] = self._bos_token
        vocab_dict['eos_token'] = self._eos_token
        vocab_dict['mask_token'] = self._mask_token
        vocab_dict['sep_token'] = self._sep_token
        vocab_dict['cls_token'] = self._cls_token
        return json.dumps(vocab_dict)

    @classmethod
    def from_json(cls, json_str):
        """Deserialize BERTVocab object from json string.

        Parameters
        ----------
        json_str : str
            Serialized json string of a BERTVocab object.

        Returns
        -------
        BERTVocab
        """
        vocab_dict = json.loads(json_str)

        unknown_token = vocab_dict.get('unknown_token')
        bert_vocab = cls(unknown_token=unknown_token)
        bert_vocab._idx_to_token = vocab_dict.get('idx_to_token')
        bert_vocab._token_to_idx = vocab_dict.get('token_to_idx')
        if unknown_token:
            bert_vocab._token_to_idx = DefaultLookupDict(bert_vocab._token_to_idx[unknown_token],
                                                         bert_vocab._token_to_idx)
        bert_vocab._reserved_tokens = vocab_dict.get('reserved_tokens')
        bert_vocab._padding_token = vocab_dict.get('padding_token')
        bert_vocab._bos_token = vocab_dict.get('bos_token')
        bert_vocab._eos_token = vocab_dict.get('eos_token')
        bert_vocab._mask_token = vocab_dict.get('mask_token')
        bert_vocab._sep_token = vocab_dict.get('sep_token')
        bert_vocab._cls_token = vocab_dict.get('cls_token')

        return bert_vocab
