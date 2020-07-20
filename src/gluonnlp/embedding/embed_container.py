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

# pylint: disable=consider-iterating-dictionary, too-many-lines
"""Load token embedding"""

__all__ = [
    'StaticEmbedding', 'FastText'
]

import io
import logging
import os
import warnings
import fasttext
import numpy as np
import mxnet as mx
from .embed_loader import load_embeddings
from .evaluation import CosineSimilarity, HyperbolicCosineSimilarity
from ..data import Vocab

class StaticEmbedding:
    def __init__(self, vocab=None, matrix=None, embedding_space="euclidean"):
        self.embedding_space = embedding_space
        self.unk_token = unk_token
        self._vocab = vocab
        self._data = matrix

    @property
    def unk_method(self):
        return None

    @property
    def data(self):
        return self._data

    @property
    def vocab(self):
        return self._vocab

    def load(self, vocab=None, pretrained_name_or_dir='glove.6B.50d', unknown='<unk>', unk_method=None):
        method = self.unk_method if self.unk_method is not None else unk_method
        if vocab is None:
            self._data, self._vocab = load_embeddings(vocab, pretrained_name_or_dir,
                                                      unknown, unk_method=method)
        else:
            self._vocab = vocab
            self._data = load_embeddings(vocab, pretrained_name_or_dir,
                                         unknown=unknown, unk_method=self.method)
        return self.data, self.vocab

    def similarity(self, words1, words2, eps=1e-10):
        squeeze = False
        if not isinstance(words1, list):
            words1 = [words1]
            squeeze = True
        if not isinstance(words2, list):
            words2 = [words2]
        if len(words1) != len(words2):
        try:
            a = self.data[self.vocab[words1]]
            b = self.data[self.vocab[words2]]
        except Exception as e:
            raise e
        a = mx.nd.array(a)
        b = mx.nd.array(b)
        if self.embedding_space == 'euclidean':
            res = CosineSimilarity(a, b, eps)
        elif self.embedding_space == 'hyperbolic':
            res = HyperbolicCosineSimilarity(a, b, eps)
        else:
            raise NotImplementedError
        if squeeze:
            res = res.squeeze()
        return res


class FastText(StaticEmbedding):
    def __init__(self, model_name_or_dir='wiki.simple', **kwargs):
        super(FastText, self).__init__(**kwargs)
        if os.path.exists(model_name_or_dir):
            file_path = model_name_or_dir
        else:
            source = model_name_or_dir
            root_path = os.path.expanduser(os.path.join(get_home_dir(), 'embedding'))
            embedding_dir = os.path.join(root_path, 'fasttext')
            if source not in C.FAST_TEXT_BIN_SHA1:
                raise ValueError('Cannot recognize {} for the bin file'.format(source))
            file_name, file_hash = C.FAST_TEXT_BIN_SHA1[source]
            file_path = _get_file_path('fasttext', file_name, file_hash)
        self._model = fasttext.load_model(file_path)

    @property
    def unk_method(self):
        return self.compute

    def compute(self, words):
        squeeze = False
        if not isinstance(words, list):
            words = [words]
            squeeze = True
        res = [self._model[word] for word in words]
        res = np.array(res)
        if squeeze:
            res = res.squeeze()
        return res

