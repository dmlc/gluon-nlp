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
from mxnet import nd, registry
from mxnet.gluon import HybridBlock, Block

__all__ = [
    'register', 'create', 'list_evaluation_functions',
    'WordEmbeddingSimilarityFunction', 'WordEmbeddingAnalogyFunction',
    'CosineSimilarity', 'ThreeCosMul', 'WordEmbeddingSimilarity',
    'WordEmbeddingAnalogy'
]


class _WordEmbeddingEvaluationFunction(HybridBlock):  # pylint: disable=abstract-method
    """Base class for word embedding evaluation functions."""
    pass


class WordEmbeddingSimilarityFunction(_WordEmbeddingEvaluationFunction):  # pylint: disable=abstract-method
    """Base class for word embedding similarity functions."""
    pass


class WordEmbeddingAnalogyFunction(_WordEmbeddingEvaluationFunction):  # pylint: disable=abstract-method
    """Base class for word embedding analogy functions.

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    k : int, default 1
        Number of analogies to predict per input triple.
    eps : float, optional, default=1e-10
        A small constant for numerical stability.
    """
    pass


###############################################################################
# Similarity and analogy functions registry helpers
###############################################################################
_REGSITRY_KIND_CLASS_MAP = {
    'similarity': WordEmbeddingSimilarityFunction,
    'analogy': WordEmbeddingAnalogyFunction
}


def register(class_):
    """Registers a new word embedding evaluation function.

    Once registered, we can create an instance with
    :func:`~gluonnlp.embedding.evaluation.create`.

    Examples
    --------
    >>> @gluonnlp.embedding.evaluation.register
    ... class MySimilarityFunction(WordEmbeddingSimilarityFunction):
    ...     def __init__(self, eps=1e-10):
    ...         pass
    >>> similarity_function = gluonnlp.embedding.create('MySimilarityFunction')
    >>> print(type(similarity_function))
    <class '__main__.MySimilarityFunction'>

    >>> @gluonnlp.embedding.evaluation.register
    ... class MyAnalogyFunction(WordEmbeddingAnalogyFunction):
    ...     def __init__(self, idx_to_vec, k=1, eps=1E-10):
    ...         pass
    >>> analogy_function = gluonnlp.embedding.create('MyAnalogyFunction')
    >>> print(type(analogy_function))
    <class '__main__.MyAnalogyFunction'>

    """

    if issubclass(class_, WordEmbeddingSimilarityFunction):
        register_ = registry.get_register_func(
            WordEmbeddingSimilarityFunction,
            'word embedding similarity evaluation function')
    elif issubclass(class_, WordEmbeddingAnalogyFunction):
        register_ = registry.get_register_func(
            WordEmbeddingAnalogyFunction,
            'word embedding analogy evaluation function')
    else:
        raise RuntimeError(
            'The custom function must either subclass '
            'WordEmbeddingSimilarityFunction or WordEmbeddingAnalogyFunction')

    return register_(class_)


def create(kind, name, **kwargs):
    """Creates an instance of a registered word embedding evaluation function.

    Parameters
    ----------
    kind : ['similarity', 'analogy', None]
        Return only valid names for similarity, analogy or both kinds of functions.
    name : str
        The evaluation function name (case-insensitive).


    Returns
    -------
    An instance of
    :class:`gluonnlp.embedding.evaluation.WordEmbeddingAnalogyFunction`:
    or
    :class:`gluonnlp.embedding.evaluation.WordEmbeddingSimilarityFunction`:
        An instance of the specified evaluation function.

    """
    if not kind in _REGSITRY_KIND_CLASS_MAP.keys():
        raise KeyError('Cannot find `kind` {}. Use '
                       '`list_sources(kind=None).keys()` to get all the valid'
                       'kinds of evaluation functions.'.format(kind))

    create_ = registry.get_create_func(
        _REGSITRY_KIND_CLASS_MAP[kind],
        'word embedding {} evaluation function'.format(kind))

    return create_(name, **kwargs)


def list_evaluation_functions(kind=None):
    """Get valid word embedding functions names.

    Parameters
    ----------
    kind : ['similarity', 'analogy', None]
        Return only valid names for similarity, analogy or both kinds of functions.

    Returns
    -------
    dict or list:
        A list of all the valid evaluation function names for the specified
        kind. If kind is set to None, returns a dict mapping each valid name to
        its respective output list. The valid names can be plugged in
        `gluonnlp.model.word_evaluation_model.create(name)`.

    """

    if kind is None:
        kind = tuple(_REGSITRY_KIND_CLASS_MAP.keys())

    if not isinstance(kind, tuple):
        if kind not in _REGSITRY_KIND_CLASS_MAP.keys():
            raise KeyError(
                'Cannot find `kind` {}. Use '
                '`list_sources(kind=None).keys()` to get all the valid'
                'kinds of evaluation functions.'.format(kind))

        reg = registry.get_registry(_REGSITRY_KIND_CLASS_MAP[kind])
        return list(reg.keys())
    else:
        return {
            embedding_name:
            list(embedding_cls.pretrained_file_name_sha1.keys())
            for embedding_name, embedding_cls in registry.get_registry(
                TokenEmbedding).items()
        }


###############################################################################
# Word embedding similarity functions
###############################################################################


@register
class CosineSimilarity(WordEmbeddingSimilarityFunction):
    """Computes the cosine similarity.

    Parameters
    ----------
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    """

    def __init__(self, eps=1e-10, **kwargs):
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


###############################################################################
# Word embedding analogy functions
###############################################################################
@register
class ThreeCosMul(WordEmbeddingAnalogyFunction):
    """The 3CosMul analogy function.

    The 3CosMul analogy function is defined as

    .. math::
        \\arg\\max_{b^* ∈ V}\\frac{\\cos(b^∗, b) \\cos(b^*, a)}{cos(b^*, a^*) + ε}

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    k : int, default 1
        Number of analogies to predict per input triple.
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    """

    def __init__(self, idx_to_vec, k=1, eps=1E-10, **kwargs):
        super(ThreeCosMul, self).__init__(**kwargs)

        self.k = k
        self.eps = eps

        self._vocab_size, self._embed_size = idx_to_vec.shape

        idx_to_vec = mx.nd.L2Normalization(idx_to_vec, eps=self.eps)
        with self.name_scope():
            self.weight = self.params.get_constant('weight', idx_to_vec)

    def hybrid_forward(self, F, words1, words2, words3, weight):  # pylint: disable=arguments-differ
        """Implement forward computation."""
        words123 = F.concat(words1, words2, words3, dim=0)
        embeddings_words123 = F.Embedding(words123, weight,
                                          input_dim=self._vocab_size,
                                          output_dim=self._embed_size)
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


@register
class ThreeCosAdd(WordEmbeddingAnalogyFunction):
    """The 3CosAdd analogy function.

    The 3CosAdd analogy function is defined as

    .. math::
        \\arg\\max_{b^* ∈ V}[\\cos(b^∗, b - a + a^*)]

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    normalize : bool, default True
        Normalize all word embeddings before computing the analogy.
    k : int, default 1
        Number of analogies to predict per input triple.
    eps : float, optional, default=1e-10
        A small constant for numerical stability.


    """

    def __init__(self, idx_to_vec, normalize=True, k=1, eps=1E-10, **kwargs):
        super(ThreeCosAdd, self).__init__(**kwargs)

        self.k = k
        self.eps = eps
        self.normalize = normalize

        self._vocab_size, self._embed_size = idx_to_vec.shape

        if self.normalize:
            idx_to_vec = mx.nd.L2Normalization(idx_to_vec, eps=self.eps)

        with self.name_scope():
            self.weight = self.params.get_constant('weight', idx_to_vec)

    def hybrid_forward(self, F, words1, words2, words3, weight):  # pylint: disable=arguments-differ
        """Implement forward computation."""
        if self.normalize:
            words123 = F.concat(words1, words2, words3, dim=0)
            embeddings_words123 = F.Embedding(words123, weight,
                                              input_dim=self._vocab_size,
                                              output_dim=self._embed_size)
            similarities = F.FullyConnected(
                embeddings_words123, weight, no_bias=True,
                num_hidden=self._vocab_size, flatten=False)
            sim_w1w4, sim_w2w4, sim_w3w4 = F.split(similarities, num_outputs=3,
                                                   axis=0)
            pred_idxs = F.topk(sim_w3w4 - sim_w1w4 + sim_w2w4, k=self.k)
        else:
            words123 = F.concat(words1, words2, words3, dim=0)
            embeddings_words123 = F.Embedding(words123, weight,
                                              input_dim=self._vocab_size,
                                              output_dim=self._embed_size)

            embeddings_word1, embeddings_word2, embeddings_word3 = F.split(
                similarities, num_outputs=3, axis=0)
            vector = (embeddings_word3 - embeddings_word1 + embeddings_word2)
            similarities = F.FullyConnected(vector, weight, no_bias=True,
                                            num_hidden=self._vocab_size,
                                            flatten=False)
            pred_idxs = F.topk(similarities, k=self.k)
        return pred_idxs


###############################################################################
# Evaluation blocks
###############################################################################
class WordEmbeddingSimilarity(HybridBlock):
    """Word embeddings similarity task evaluator.

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    similarity_function : str, default 'CosineSimilarity'
        Name of a registered WordEmbeddingSimilarityFunction.
    eps : float, optional, default=1e-10
        A small constant for numerical stability.

    """

    def __init__(self, idx_to_vec, similarity_function='CosineSimilarity',
                 eps=1e-10, **kwargs):
        super(WordEmbeddingSimilarity, self).__init__(**kwargs)

        self.eps = eps
        self._vocab_size, self._embed_size = idx_to_vec.shape

        with self.name_scope():
            self.weight = self.params.get_constant('weight', idx_to_vec)
            self.similarity = create(kind='similarity',
                                     name=similarity_function, eps=self.eps)

        if not isinstance(self.similarity, WordEmbeddingSimilarityFunction):
            raise RuntimeError(
                '{} is not a WordEmbeddingAnalogyFunction'.format(name))

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


class WordEmbeddingAnalogy(Block):
    """Word embeddings analogy task evaluator.

    Parameters
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        Embedding matrix.
    analogy_function : str, default 'ThreeCosMul'
        Name of a registered WordEmbeddingAnalogyFunction.
    k : int, default 1
        Number of analogies to predict per input triple.
    exclude_question_words : bool (True)
        Exclude the 3 question words from being a valid answer.

    """

    def __init__(self, idx_to_vec, analogy_function='ThreeCosMul', k=1,
                 exclude_question_words=True, **kwargs):
        super(WordEmbeddingAnalogy, self).__init__(**kwargs)

        assert k >= 1
        self.k = k
        self.exclude_question_words = exclude_question_words

        self._internal_k = self.k + 3 * self.exclude_question_words

        with self.name_scope():
            self.analogy = create(kind='analogy', name=analogy_function,
                                  idx_to_vec=idx_to_vec, k=self._internal_k)

        if not isinstance(self.analogy, WordEmbeddingAnalogyFunction):
            raise RuntimeError(
                '{} is not a WordEmbeddingAnalogyFunction'.format(name))

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

        if self.exclude_question_words:
            orig_context = pred_idxs.context
            pred_idxs = pred_idxs.asnumpy().tolist()
            pred_idxs = [[
                idx for idx in row
                if idx != w1 and idx != w2 and idx != w3
            ] for row, w1, w2, w3 in zip(pred_idxs, words1, words2, words3)]
            pred_idxs = [p[:self.k] for p in pred_idxs]
            pred_idxs = nd.array(pred_idxs, ctx=orig_context)

        return pred_idxs
