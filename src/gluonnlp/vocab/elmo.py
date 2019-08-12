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
"""Vocabulary class used in the original pre-trained ELMo models."""

# pylint: disable=consider-iterating-dictionary

__all__ = ['ELMoCharVocab']

class ELMoCharVocab:
    r"""ELMo special character vocabulary

    The vocab aims to map individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here.

    Specifically, char ids 0-255 come from utf-8 encoding bytes.
    Above 256 are reserved for special tokens.

    Parameters
    ----------
    bos_token : hashable object or None, default '<bos>'
        The representation for the special token of beginning-of-sequence token.
    eos_token : hashable object or None, default '<eos>'
        The representation for the special token of end-of-sequence token.

    Attributes
    ----------
    max_word_length : 50
        The maximum number of character a word contains is 50 in ELMo.
    bos_id : 256
        The index of beginning of the sentence character is 256 in ELMo.
    eos_id : 257
        The index of end of the sentence character is 257 in ELMo.
    bow_id : 258
        The index of beginning of the word character is 258 in ELMo.
    eow_id : 259
        The index of end of the word character is 259 in ELMo.
    pad_id : 260
        The index of padding character is 260 in ELMo.
    """
    max_word_length = 50
    max_word_chars = 48 # excluding bow and eow
    # char ids 0-255 come from utf-8 encoding bytes
    bos_id = 256
    eos_id = 257
    bow_id = 258
    eow_id = 259
    pad_id = 260

    def __init__(self, bos_token='<bos>', eos_token='<eos>'):
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._id_dict = {bos_token: [ELMoCharVocab.bos_id],
                         eos_token: [ELMoCharVocab.eos_id]}

    def __getitem__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.

        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.

        Returns
        -------
        int or list of ints
            A list of char indices or a list of list of char indices according to the vocabulary.
        """

        if not isinstance(tokens, (list, tuple)):
            return self._token_to_char_indices(tokens)
        else:
            return [self._token_to_char_indices(token) for token in tokens]

    def _token_to_char_indices(self, token):
        ids = [ELMoCharVocab.pad_id] * ELMoCharVocab.max_word_length
        ids[0] = ELMoCharVocab.bow_id
        word_ids = bytearray(token, 'utf-8', 'ignore')[:ELMoCharVocab.max_word_chars]
        word_ids = self._id_dict.get(token, word_ids)
        ids[1:(1+len(word_ids))] = word_ids
        ids[1+len(word_ids)] = ELMoCharVocab.eow_id
        return ids

    def __call__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A list of char indices or a list of list of char indices according to the vocabulary.
        """

        return self[tokens]

    def __len__(self):
        return 262
