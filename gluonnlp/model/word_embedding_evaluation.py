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
# pylint: disable=eval-used, redefined-outer-name
"""Models for intrinsic and extrinsic word embedding evaluation"""

import os
import warnings

from mxnet.gluon.model_zoo.model_store import get_model_file
from mxnet import init, nd, cpu, autograd
from mxnet.gluon import nn, Block, HybridBlock
from mxnet.gluon.model_zoo import model_store

from .utils import _get_rnn_layer
from .utils import apply_weight_drop
from ..data.utils import _load_pretrained_vocab


class CosineSimilarity(HybridBlock):
    """Computes the cosine similarity."""

    def hybrid_forward(self, F, x, y):
        """Implement forward computation."""
        x = F.L2Normalization(x)
        y = F.L2Normalization(y)
        x = F.expand_dims(x, axis=1)
        y = F.expand_dims(y, axis=2)
        return F.batch_dot(x, y).reshape((-1, ))


class WordEmbeddingSimilarity(HybridBlock):
    """Helper class to evaluate word embeddings based on similarity task.

    The Evaluator must be initialized, giving the option to adapt the
    parameters listed below. An Evaluator object can be called with the
    signature defined at Call Signature.

    Parameters
    ----------
    metric : mx.metric.EvalMetric
        Metric for computing the overall score given the list of predicted
        similarities and ground truth similarities. Defaults to
        SpearmanRankCorrelation.
    similarity_function : function
        Given two mx.nd.NDArray's of shape (dataset_size, embedding_size),
        compute a similarity score.

    Call Signature
    --------------
    token_embedding : gluonnlp.embedding.TokenEmbedding
        Embedding to evaluate.
    dataset : mx.gluon.Dataset
        Dataset consisting of rows with 3 elements: [word1, word2, score]

    Reference: https://github.com/salesforce/awd-lstm-lm

    License: BSD 3-Clause

    Parameters
    ----------
    """

    def __init__(self, vocab_size, embed_size, similarity=CosineSimilarity(),
                 embedding_params=None, **kwargs):
        super(WordEmbeddingSimilarity, self).__init__(**kwargs)

        self.similarity = similarity
        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size, embed_size,
                                          weight_initializer=init.Uniform(0.1),
                                          params=embedding_params)

    def hybrid_forward(self, F, words1, words2):  # pylint: disable=arguments-differ
        """Implement forward computation.

        Parameters
        ----------
        inputs : NDArray
            The training dataset.
        begin_state : list
            The initial hidden states.

        Returns
        -------
        similarity : NDArray
            The output of the model.
        """
        embeddings_words1 = self.embedding(words1)
        embeddings_words2 = self.embedding(words2)
        similarity = self.similarity(embeddings_words1, embeddings_words2)
        return similarity

