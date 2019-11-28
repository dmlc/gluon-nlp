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
# pylint: disable=invalid-encoded-data, too-many-lines
"""Transformer API.

It provides tools for common transformation on samples in text dataset, such as
clipping, padding, and tokenization.
"""
import unicodedata
from typing import List, Optional

import gluonnlp as nlp

__all__ = ['XLNetTokenizer']


class XLNetTokenizer:
    """End-to-end tokenization for XLNet models.

    Parameters
    ----------
    sentencepiece_path
        Path to sentencepiece model, to be used for obtaining word pieces.

    .. note::

        For multi-processing, making an extra copy of the XLNetTokenizer instance
        is recommended before calling it for the first time is recommended.
        SentencePiece models can't be pickled, which is needed for
        multi-processing. The SentencePiece model is initialized during the first
        call.

    Examples
    --------
    >>> _, vocab = gluonnlp.model.bert_12_768_12(dataset_name='wiki_multilingual_uncased',
    ...                                          pretrained=False, root='./model')
    -etc-
    >>> tokenizer = gluonnlp.data.BERTTokenizer(vocab=vocab)
    >>> tokenizer('gluonnlp: 使NLP变得简单。')
    ['gl', '##uo', '##nn', '##lp', ':', '使', 'nl', '##p', '变', '得', '简', '单', '。']

    """
    _spiece_prefix = '▁'

    def __init__(self, sentencepiece_path: str, lower: bool = False, remove_space: bool = True,
                 keep_accents: bool = False):
        self._sentencepiece_path = sentencepiece_path
        self._lower = lower
        self._remove_space = remove_space
        self._keep_accents = keep_accents
        self._sentencepiece = None  # type: Optional[nlp.data.SentencepieceTokenizer]

    def __call__(self, sample: str) -> List[str]:
        """Tokenize a sample.

        Parameters
        ----------
        sample
            The string to tokenize.

        Returns
        -------
        tokens
            List of tokens
        """

        if self._remove_space:
            outputs = ' '.join(sample.strip().split())
        else:
            outputs = sample
        outputs = outputs.replace('``', '"').replace('\'\'', '"')

        if not self._keep_accents:
            outputs = unicodedata.normalize('NFKD', outputs)
            outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
        if self._lower:
            outputs = outputs.lower()

        if self._sentencepiece is None:
            self._sentencepiece = nlp.data.SentencepieceTokenizer(self._sentencepiece_path)

        pieces = self._sentencepiece(outputs)
        new_pieces = []  # type: List[str]
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self._sentencepiece(piece[:-1].replace(self._spiece_prefix, ''))
                if piece[0] != self._spiece_prefix and cur_pieces[0][0] == self._spiece_prefix:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces
