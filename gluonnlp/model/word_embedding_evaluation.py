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

import mxnet as mx
from mxnet import nd
from mxnet.gluon import HybridBlock, Block


class CosineSimilarity(HybridBlock):
    """Computes the cosine similarity.

    Parameters
    ----------
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    """

    def __init__(self, eps=1e-10):
        super(CosineSimilarity, self).__init__(**kwargs)
        self.eps = eps

    def hybrid_forward(self, F, x, y):  # pylint: disable=arguments-differ
        """Compute the cosine similarity between two batches of vectors.

        The cosine similarity is the dot product between the L2 normalized
        vectors.

        Parameters
        ----------
        x : Symbol or NDArray
        y : Symbol or NDArray

        Returns
        -------
        similarity : Symbol or NDArray
            The similarity computed by WordEmbeddingSimilarity.similarity_function.

        """

        x = F.L2Normalization(x, eps=self.eps)
        y = F.L2Normalization(y, eps=self.eps)
        x = F.expand_dims(x, axis=1)
        y = F.expand_dims(y, axis=2)
        return F.batch_dot(x, y).reshape((-1, ))


class WordEmbeddingSimilarity(HybridBlock):
    """Word embeddings similarity task evaluator.

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    similarity_function : mxnet.gluon.Block
        Given two mx.nd.NDArray's of word embeddings compute a similarity
        score.
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    """

    def __init__(self, idx_to_vec, similarity_function=CosineSimilarity,
                 eps=1e-10, **kwargs):
        super(WordEmbeddingSimilarity, self).__init__(**kwargs)

        self.eps = eps
        self._vocab_size, self._embed_size = idx_to_vec.shape

        with self.name_scope():
            self.weight = self.params.get_constant('weight', idx_to_vec)
            self.similarity = similarity_function(eps=self.eps)

    def hybrid_forward(self, F, words1, words2, weight):  # pylint: disable=arguments-differ
        """Predict the similarity of words1 and words2.

        Parameters
        ----------
        words1 : Symbol or NDArray
            The indices of the words the we wish to compare to the words in words2.
        words2 : Symbol or NDArray
            The indices of the words the we wish to compare to the words in words1.

        Returns
        -------
        similarity : Symbol or NDArray
            The similarity computed by WordEmbeddingSimilarity.similarity_function.
        """
        embeddings_words1 = F.Embedding(words1, weight,
                                        input_dim=self._vocab_size,
                                        output_dim=self._embed_size)
        embeddings_words2 = F.Embedding(words2, weight,
                                        input_dim=self._vocab_size,
                                        output_dim=self._embed_size)
        similarity = self.similarity(embeddings_words1, embeddings_words2)
        return similarity


class ThreeCosMul(HybridBlock):
    """The 3CosMul analogy function.

    The 3CosMul analogy function is defined as
    .. math::
        \\arg\\max_{b^* ∈ V}\\frac{\\cos(b^∗, b) \\cos(b^*, a)}{cos(b^*, a^*) + ε}

    Parameters
    ----------
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    """

    def __init__(self, idx_to_vec, k=1, eps=1E-10, **kwargs):
        super(ThreeCosMul, self).__init__(**kwargs)

        self.k = k
        self.eps = eps

        self._vocab_size, self._embed_size = idx_to_vec.shape

        normalized_idx_to_vec = mx.nd.L2Normalization(idx_to_vec, eps=self.eps)
        with self.name_scope():
            self.weight = self.params.get_constant('weight',
                                                   normalized_idx_to_vec)

    def hybrid_forward(self, F, words1, words2, words3, weight):  # pylint: disable=arguments-differ
        """Implement forward computation."""
        embeddings_words1 = F.Embedding(words1, weight,
                                        input_dim=self._vocab_size,
                                        output_dim=self._embed_size)
        embeddings_words2 = F.Embedding(words2, weight,
                                        input_dim=self._vocab_size,
                                        output_dim=self._embed_size)
        embeddings_words3 = F.Embedding(words3, weight,
                                        input_dim=self._vocab_size,
                                        output_dim=self._embed_size)
        embeddings_words123 = F.concat(embeddings_words1, embeddings_words2,
                                       embeddings_words3, dim=0)

        similarities = F.FullyConnected(
            embeddings_words123, weight, no_bias=True,
            num_hidden=self._vocab_size, flatten=False)
        # Map cosine similarities to [0, 1]
        similarities = (similarities + 1) / 2

        sim_w1w4, sim_w2w4, sim_w3w4 = F.split(similarities, num_outputs=3,
                                               axis=0)

        pred_idxs = F.topk((sim_w2w4 * sim_w3w4) / (sim_w1w4 + self.eps),
                           k=self.k)
        return pred_idxs


class WordEmbeddingAnalogy(Block):
    """Word embeddings analogy task evaluator.

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    analogy_function : mxnet.gluon.Block
        Given three mx.nd.NDArray's of word embeddings predict k analogies.
        idx_to_vec and k are passed on to its constructor.
    k : int (1)
        Number of analogies to predict per input triple.
    exclude_inputs : bool (True)
        Exclude the 3 input words from being a valid analogy output.

    """

    def __init__(self, idx_to_vec, analogy_function=ThreeCosMul, k=1,
                 exclude_inputs=True, **kwargs):
        super(WordEmbeddingAnalogy, self).__init__(**kwargs)

        assert k >= 1
        self.k = k
        self.exclude_inputs = exclude_inputs

        self._internal_k = self.k + 3 * self.exclude_inputs

        with self.name_scope():
            self.analogy = analogy_function(idx_to_vec=idx_to_vec,
                                            k=self._internal_k)

    def forward(self, words1, words2, words3):  # pylint: disable=arguments-differ
        """Implement forward computation.

        Parameters
        ----------
        words1 : NDArray
            Word indices.
        words2 : NDArray
            Word indices.
        words3 : NDArray
            Word indices.

        Returns
        -------
        predicted_indices : NDArray
            Predicted indices of shape (batch_size, k)
        """
        pred_idxs = self.analogy(words1, words2, words3)

        if self.exclude_inputs:
            orig_context = pred_idxs.context
            pred_idxs = pred_idxs.asnumpy().tolist()
            pred_idxs = [[
                idx for i, idx in enumerate(row)
                if idx != w1 and idx != w2 and idx != w3
            ] for row, w1, w2, w3 in zip(pred_idxs, words1, words2, words3)]
            pred_idxs = [p[:self.k] for p in pred_idxs]
            pred_idxs = nd.array(pred_idxs, ctx=orig_context)

        return pred_idxs
