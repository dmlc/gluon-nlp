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
from __future__ import absolute_import, print_function

import json
import os

from ..data.transforms import SentencepieceTokenizer
from ..data.utils import _convert_to_unicode, count_tokens
from .vocab import Vocab

__all__ = ['BERTVocab']


UNKNOWN_TOKEN = '[UNK]'
PADDING_TOKEN = '[PAD]'
MASK_TOKEN = '[MASK]'
SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'


class BERTVocab(Vocab):
    """Specialization of gluonnlp.Vocab for BERT models.

    BERTVocab changes default token representations of unknown and other
    special tokens of gluonnlp.Vocab and adds convenience parameters to specify
    mask, sep and cls tokens typically used by Bert models.

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
    unknown_token : hashable object or None, default '[UNK]'
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation. If None, looking up an unknown token will result in KeyError.
    padding_token : hashable object or None, default '[PAD]'
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
    reserved_tokens : list of hashable objects or None, default None
        A list specifying additional tokens to be added to the vocabulary.
        `reserved_tokens` cannot contain `unknown_token` or duplicate reserved
        tokens.
        Keys of `counter`, `unknown_token`, and values of `reserved_tokens`
        must be of the same hashable type. Examples of hashable types are str,
        int, and tuple.
    token_to_idx : dict mapping tokens (hashable objects) to int or None, default None
        Optionally specifies the indices of tokens to be used by the
        vocabulary. Each token in `token_to_index` must be part of the Vocab
        and each index can only be associated with a single token.
        `token_to_idx` is not required to contain a mapping for all tokens. For
        example, it is valid to only set the `unknown_token` index to 10 (instead
        of the default of 0) with `token_to_idx = {'<unk>': 10}`.

    Attributes
    ----------
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

    def __init__(self, counter=None, max_size=None, min_freq=1, unknown_token=UNKNOWN_TOKEN,
                 padding_token=PADDING_TOKEN, bos_token=None, eos_token=None, mask_token=MASK_TOKEN,
                 sep_token=SEP_TOKEN, cls_token=CLS_TOKEN, reserved_tokens=None, token_to_idx=None):

        super(BERTVocab, self).__init__(counter=counter, max_size=max_size, min_freq=min_freq,
                                        unknown_token=unknown_token, padding_token=padding_token,
                                        bos_token=bos_token, eos_token=eos_token,
                                        reserved_tokens=reserved_tokens, cls_token=cls_token,
                                        sep_token=sep_token, mask_token=mask_token,
                                        token_to_idx=token_to_idx)

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
        token_to_idx = vocab_dict.get('token_to_idx')
        unknown_token = vocab_dict.get('unknown_token')
        reserved_tokens = vocab_dict.get('reserved_tokens')
        identifiers_to_tokens = vocab_dict.get('identifiers_to_tokens', dict())

        special_tokens = {unknown_token}

        # Backward compatibility for explicit serialization of padding_token,
        # bos_token, eos_token, mask_token, sep_token, cls_token handling in
        # the json string as done in older versions of GluonNLP.
        deprecated_arguments = [
            'padding_token', 'bos_token', 'eos_token', 'mask_token', 'sep_token', 'cls_token'
        ]
        for token_name in deprecated_arguments:
            token = vocab_dict.get(token_name)
            if token is not None:
                assert token_name not in identifiers_to_tokens, 'Invalid json string. ' \
                    '{} was serialized twice.'.format(token_name)
                identifiers_to_tokens[token_name] = token

        # Separate reserved from special tokens
        special_tokens.update(identifiers_to_tokens.values())
        if reserved_tokens is not None:
            reserved_tokens = [
                t for t in reserved_tokens if t not in special_tokens
            ]

        return cls(counter=count_tokens(token_to_idx.keys()),
                   unknown_token=unknown_token,
                   reserved_tokens=reserved_tokens,
                   token_to_idx=token_to_idx,
                   **identifiers_to_tokens)

    @classmethod
    def from_sentencepiece(cls,
                           path,
                           mask_token=MASK_TOKEN,
                           sep_token=SEP_TOKEN,
                           cls_token=CLS_TOKEN,
                           unknown_token=None,
                           padding_token=None,
                           bos_token=None,
                           eos_token=None,
                           reserved_tokens=None):
        """BERTVocab from pre-trained sentencepiece Tokenizer

        Parameters
        ----------
        path : str
            Path to the pre-trained subword tokenization model.
        mask_token : hashable object or None, default '[MASK]'
            The representation for the special token of mask token for BERT
        sep_token : hashable object or None, default '[SEP]'
            a token used to separate sentence pairs for BERT.
        cls_token : hashable object or None, default '[CLS]'
        unknown_token : hashable object or None, default None
            The representation for any unknown token. In other words,
            any unknown token will be indexed as the same representation.
            If set to None, it is set to the token corresponding to the unk_id()
            in the loaded sentencepiece model.
        padding_token : hashable object or None, default '[PAD]'
            The representation for padding token.
        bos_token : hashable object or None, default None
            The representation for the begin of sentence token.
            If set to None, it is set to the token corresponding to the bos_id()
            in the loaded sentencepiece model.
        eos_token : hashable object or None, default None
            The representation for the end of sentence token.
            If set to None, it is set to the token corresponding to the bos_id()
            in the loaded sentencepiece model.
        reserved_tokens : list of strs or None, optional
            A list of reserved tokens that will always be indexed.

        Returns
        -------
        BERTVocab
        """
        sp = SentencepieceTokenizer(os.path.expanduser(path))
        processor = sp._processor

        # we manually construct token_to_idx, idx_to_token and relevant fields for a BERT vocab.
        token_to_idx = {
            _convert_to_unicode(t): i
            for i, t in enumerate(sp.tokens)
        }

        def _check_consistency(processor, token_id, provided_token):
            """Check if provided_token is consistent with the special token inferred
            from the loaded sentencepiece vocab."""
            provided_token = _convert_to_unicode(provided_token) if provided_token else None
            if token_id >= 0:
                # sentencepiece contains this special token.
                token = _convert_to_unicode(processor.IdToPiece(token_id))
                if provided_token:
                    assert provided_token == token
                provided_token = token
            return provided_token

        unknown_token = _check_consistency(processor, processor.unk_id(), unknown_token)
        bos_token = _check_consistency(processor, processor.bos_id(), bos_token)
        eos_token = _check_consistency(processor, processor.eos_id(), eos_token)
        padding_token = _check_consistency(processor, processor.pad_id(), padding_token)

        return cls(counter=count_tokens(token_to_idx.keys()),
                   unknown_token=unknown_token,
                   padding_token=padding_token,
                   bos_token=bos_token,
                   eos_token=eos_token,
                   mask_token=mask_token,
                   sep_token=sep_token,
                   cls_token=cls_token,
                   reserved_tokens=reserved_tokens,
                   token_to_idx=token_to_idx)
