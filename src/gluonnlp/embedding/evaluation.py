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
"""Functions for intrinsic and extrinsic word embedding evaluation"""

import mxnet as mx
import numpy as np
from ..op import l2_normalize

__all__ = ['CosineSimilarity', 'ThreeCosMul', 'ThreeCosAdd', 'HyperbolicCosineSimilarity']

def _np_batch_dot(x, y):
    return np.einsum("ij, ij -> i", x, y)

def CosineSimilarity(x, y, eps=1e-10):
    """Computes the cosine similarity.

    Parameters
    ----------
    x : NDArray
    y : NDArray
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    Returns
    -------
    similarity : numpy.ndarray
        The similarity scores between each word in x and each word in y.

    """
    dim = x.shape[-1]
    assert dim == y.shape[-1], "Embedding dim for x, y are different!"
    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)
    x = l2_normalize(mx.nd, x, eps=eps).asnumpy()
    y = l2_normalize(mx.nd, y, eps=eps).asnumpy()
    res = _np_batch_dot(x, y)
    return res.reshape((-1, ))

def _HyperbolicDist(a, b, eps):
    return 1 + 2 * _np_batch_dot(a - b, a - b)/ \
            (np.clip(1 - _np_batch_dot(a, a), eps, 1) * np.clip(1 - _np_batch_dot(b, b), eps, 1))

def HyperbolicCosineSimilarity(x, y, eps=1e-10):
    """Computes the cosine similarity in the Hyperbolic space.

    Parameters
    ----------
    x : NDArray
    y : NDArray

    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    Returns
    -------
    similarity : numpy.ndarray
        The similarity scores between each word in x and each word in y.

    """
    dim = x.shape[-1]
    assert dim == y.shape[-1], "Embedding dim for x, y are different!"
    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)
    x = l2_normalize(mx.nd, x, eps=eps).asnumpy()
    y = l2_normalize(mx.nd, y, eps=eps).asnumpy()

    xy = _HyperbolicDist(x, y, eps)
    dis_x = _HyperbolicDist(x, np.zeros_like(x), eps)
    dis_y = _HyperbolicDist(y, np.zeros_like(y), eps)
    cos_xy = (dis_x * dis_y - xy) / \
        (np.sinh(np.arccosh(dis_x)) * np.sinh(np.arccosh(dis_y)) + eps)
    return cos_xy.reshape((-1, ))

class ThreeCosMul:
    """The 3CosMul analogy function.

    The 3CosMul analogy function is defined as

    .. math::
        \\arg\\max_{b^* ∈ V}\\frac{\\cos(b^∗, b) \\cos(b^*, a)}{cos(b^*, a^*) + ε}

    See the following paper for more details:

    - Levy, O., & Goldberg, Y. (2014). Linguistic regularities in sparse and
      explicit word representations. In R. Morante, & W. Yih, Proceedings of the
      Eighteenth Conference on Computational Natural Language Learning, CoNLL 2014,
      Baltimore, Maryland, USA, June 26-27, 2014 (pp. 171–180). : ACL.

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    k : int, default 1
        Number of analogies to predict per input triple.
    exclude_question_words : bool, default True
        Exclude the 3 question words from being a valid answer.
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    """

    def __init__(self, idx_to_vec, k=1, eps=1E-10, exclude_question_words=True):
        self.k = k
        self.eps = eps
        self._exclude_question_words = exclude_question_words

        self._vocab_size, self._embed_size = idx_to_vec.shape

        idx_to_vec = mx.nd.L2Normalization(idx_to_vec, eps=self.eps)
        self.weight = idx_to_vec

    def compute(self, words1, words2, words3):  # pylint: disable=arguments-differ
        """Compute ThreeCosMul for given question words.

        Parameters
        ----------
        words1 : NDArray
            Question words at first position. Shape (batch_size, )
        words2 : NDArray
            Question words at second position. Shape (batch_size, )
        words3 : NDArray
            Question words at third position. Shape (batch_size, )

        Returns
        -------
        NDArray
            Predicted answer words. Shape (batch_size, k).

        """
        words123 = mx.nd.concat(words1, words2, words3, dim=0)
        embeddings_words123 = mx.nd.Embedding(words123, self.weight,
                                              input_dim=self._vocab_size,
                                              output_dim=self._embed_size)
        similarities = mx.nd.FullyConnected(
            embeddings_words123, self.weight, no_bias=True,
            num_hidden=self._vocab_size, flatten=False)
        # Map cosine similarities to [0, 1]
        similarities = (similarities + 1) / 2

        sim_w1w4, sim_w2w4, sim_w3w4 = mx.nd.split(similarities, num_outputs=3, axis=0)

        sim = (sim_w2w4 * sim_w3w4) / (sim_w1w4 + self.eps)

        if self._exclude_question_words:
            for words in [words1, words2, words3]:
                sim = sim * mx.nd.one_hot(words, self.weight.shape[0], 0, 1)

        pred_idxs = mx.nd.topk(sim, k=self.k)
        return pred_idxs

class ThreeCosAdd:
    """The 3CosAdd analogy function.

    The 3CosAdd analogy function is defined as

    .. math::
        \\arg\\max_{b^* ∈ V}[\\cos(b^∗, b - a + a^*)]

    See the following paper for more details:

    - Levy, O., & Goldberg, Y. (2014). Linguistic regularities in sparse and
      explicit word representations. In R. Morante, & W. Yih, Proceedings of the
      Eighteenth Conference on Computational Natural Language Learning, CoNLL 2014,
      Baltimore, Maryland, USA, June 26-27, 2014 (pp. 171–180). : ACL.

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    normalize : bool, default True
        Normalize all word embeddings before computing the analogy.
    k : int, default 1
        Number of analogies to predict per input triple.
    exclude_question_words : bool, default True
        Exclude the 3 question words from being a valid answer.
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    """

    def __init__(self, idx_to_vec, normalize=True, k=1, eps=1e-10, exclude_question_words=True):
        self.k = k
        self.eps = eps
        self.normalize = normalize
        self._exclude_question_words = exclude_question_words
        self._vocab_size, self._embed_size = idx_to_vec.shape

        if self.normalize:
            idx_to_vec = mx.nd.L2Normalization(idx_to_vec, eps=self.eps)

        self.weight = idx_to_vec

    def compute(self, words1, words2, words3): # pylint: disable=arguments-differ
        """Compute ThreeCosAdd for given question words.

        Parameters
        ----------
        words1 : NDArray
            Question words at first position. Shape (batch_size, )
        words2 : NDArray
            Question words at second position. Shape (batch_size, )
        words3 : NDArray
            Question words at third position. Shape (batch_size, )

        Returns
        -------
        NDArray
            Predicted answer words. Shape (batch_size, k).

        """
        words123 = mx.nd.concat(words1, words2, words3, dim=0)
        embeddings_words123 = mx.nd.Embedding(words123, self.weight,
                                              input_dim=self._vocab_size,
                                              output_dim=self._embed_size)
        if self.normalize:
            similarities = mx.nd.FullyConnected(
                embeddings_words123, self.weight, no_bias=True,
                num_hidden=self._vocab_size, flatten=False)
            sim_w1w4, sim_w2w4, sim_w3w4 = mx.nd.split(similarities, num_outputs=3, axis=0)
            pred = sim_w3w4 - sim_w1w4 + sim_w2w4
        else:
            embeddings_word1, embeddings_word2, embeddings_word3 = mx.nd.split(
                embeddings_words123, num_outputs=3, axis=0)
            vector = embeddings_word3 - embeddings_word1 + embeddings_word2
            pred = mx.nd.FullyConnected(
                vector, self.weight, no_bias=True, num_hidden=self._vocab_size,
                flatten=False)

        if self._exclude_question_words:
            for words in [words1, words2, words3]:
                pred = pred * mx.nd.one_hot(words, self.weight.shape[0], 0, 1)

        pred_idxs = mx.nd.topk(pred, k=self.k)
        return pred_idxs
