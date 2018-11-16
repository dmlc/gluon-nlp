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

"""Tokenizer for SQuAD dataset"""

import re
import nltk


class BiDAFTokenizer:
    """Tokenizer that is used for preprocessing data for BiDAF model. It applies basic tokenizer
    and some extra preprocessing steps making data ready to be used for training BiDAF
    """
    def __call__(self, sample, lower_case=False):
        """Process single record

        Parameters
        ----------
        sample: str
            The record to tokenize

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        sample = sample.replace('\n', ' ').replace(u'\u000A', '').replace(u'\u00A0', '')

        tokens = [token.replace('\'\'', '"').replace('``', '"') for token in
                  nltk.word_tokenize(sample)]
        tokens = BiDAFTokenizer._process_tokens(tokens)
        tokens = [token for token in tokens if len(token) > 0]

        if lower_case:
            tokens = [token.lower() for token in tokens]

        return tokens

    @staticmethod
    def _process_tokens(temp_tokens):
        """Process tokens by splitting them if split symbol is encountered

        Parameters
        ----------
        temp_tokens: list[str]
            Tokens to be processed

        Returns
        -------
        tokens : list[str]
            List of updated tokens
        """
        tokens = []
        splitters = ('-', '\u2212', '\u2014', '\u2013', '/', '~', '"', '\'', '\u201C',
                     '\u2019', '\u201D', '\u2018', '\u00B0')

        for token in temp_tokens:
            tokens.extend(re.split('([{}])'.format(''.join(splitters)), token))

        return tokens
