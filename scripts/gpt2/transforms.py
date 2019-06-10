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

"""Transforms used in GPT models."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import io
import json
import regex as re


class GPT2Tokenizer(object):
    def __init__(self, bpe_ranks_path):
        """

        Parameters
        ----------
        bpe_ranks_path : str
            Path to the BPE rank file
        """
        with io.open(bpe_ranks_path, 'r', encoding='utf-8') as f:
            bpe_data = f.read()
            self._bpe_ranks = dict()
            for i, merge_str in enumerate(bpe_data.split('\n')[1:-1]):
                self._bpe_ranks[tuple(merge_str.split())] = i
        self._cache = {}
        self._byte_encoder = self.init_byte_encoder()
        self._token_pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    @staticmethod
    def init_byte_encoder():
        codes = list(range(ord("!"), ord("~") + 1)) +\
                list(range(ord("¡"), ord("¬") + 1)) +\
                list(range(ord("®"), ord("ÿ") + 1))
        byte_encoder = {code: chr(code) for code in codes}
        shift = 0
        for code in range(2 ** 8):
            if code not in byte_encoder:
                byte_encoder[code] = chr(2 ** 8 + shift)
                shift += 1
        return byte_encoder

    #TODO(sxjscience) Use numba to accelerate the code
    def get_bpe_subword(self, token):
        """ Encode the word token into BPE subwords

        Parameters
        ----------
        token : str

        Returns
        -------
        chars : list(str)
        """
        if token in self._cache:
            return self._cache[token]
        chars = list(token)
        while len(chars) > 0:
            min_pair, min_rank = None, float('inf')
            # Find the pair with the minimum rank
            for i in range(1, len(chars)):
                pair = (chars[i - 1], chars[i])
                rank = self._bpe_ranks.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None or min_pair not in self._bpe_ranks:
                break
            # Merge the pair
            last, tail = chars[0], 1
            for index in range(1, len(chars)):
                if (last, chars[index]) == min_pair:
                    chars[tail - 1] = last + chars[index]
                    last = last + chars[index]
                else:
                    chars[tail - 1] = last
                    tail += 1
                    last = chars[index]
            chars[tail - 1] = last
            chars = chars[:tail]
        self._cache[token] = chars
        return chars

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : str

        Returns
        -------
        ret : list(str)
        """
        ret = []
        for word_token in re.findall(self._token_pattern, sample):
            word_token = bytearray(word_token.encode('utf-8'))
            word_token = ''.join(self._byte_encoder[code] for code in word_token)
            ret.extend(self.get_bpe_subword(word_token))
        return ret


class GPT2Detokenizer(object):
    def __init__(self, tokenizer):
        """

        Parameters
        ----------
        tokenizer : GPT2Tokenizer
        """
        self._byte_decoder = {v: k for k, v in tokenizer._byte_encoder.items()}

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : list(str)

        Returns
        -------
        ret : str
        """
        text = ''.join(sample)
        ret = bytearray([self._byte_decoder[byte] for byte in text]).decode('utf-8', errors='replace')
        return ret
