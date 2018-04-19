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

from mxnet import init, nd
from mxnet.gluon import HybridBlock, nn


class CosineSimilarity(HybridBlock):
    """Computes the cosine similarity."""

    def hybrid_forward(self, F, x, y):  # pylint: disable=arguments-differ
        """Implement forward computation."""
        x = F.L2Normalization(x)
        y = F.L2Normalization(y)
        x = F.expand_dims(x, axis=1)
        y = F.expand_dims(y, axis=2)
        return F.batch_dot(x, y).reshape((-1, ))


class WordEmbeddingSimilarity(HybridBlock):
    """Word embeddings similarity task evaluator.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_size : int
        Dimensionality of the embeddings.
    similarity_function : mxnet.gluon.Block
        Given two mx.nd.NDArray's of word embeddings compute a similarity
        score.
    embeddings_params : mx.gluon.ParameterDict (optional)
        Parameters of the internal WordEmbeddingSimilarity.embedding Block.

    """

    def __init__(self, vocab_size, embed_size,
                 similarity_function=CosineSimilarity, embedding_params=None,
                 **kwargs):
        super(WordEmbeddingSimilarity, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.similarity_function = similarity_function

        with self.name_scope():
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                          weight_initializer=init.Uniform(0.1),
                                          params=embedding_params)
            self.similarity = similarity_function()

    def hybrid_forward(self, F, words1, words2):  # pylint: disable=arguments-differ
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
        embeddings_words1 = self.embedding(words1)
        embeddings_words2 = self.embedding(words2)
        similarity = self.similarity(embeddings_words1, embeddings_words2)
        return similarity


class NormalizedDense(HybridBlock):
    r"""Densely-connected NN layer that L2 normalizes it's weights and inputs.

    `NormalizedDense` implements the operation: `output =
    activation(dot(L2Normalization(input), L2Normalization(weight)) + bias)`
    where `activation` is the element-wise activation function passed as the
    `activation` argument, `weight` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer (only applicable if
    `use_bias` is `True`).

    Note: the input must be a tensor with rank 2. Use `flatten` to convert it
    to rank 2 manually if necessary.

    Parameters
    ----------
    units : int
        Dimensionality of the output space.
    activation : str
        Activation function to use. See help on `Activation` layer.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias : bool
        Whether the layer uses a bias vector.
    flatten: bool
        Whether the input tensor should be flattened.
        If true, all but the first axis of input data are collapsed together.
        If false, all but the last axis of input data are kept the same, and the transformation
        applies on the last axis.
    weight_initializer : str or `Initializer`
        Initializer for the `kernel` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    in_units : int, optional
        Size of the input data. If not specified, initialization will be
        deferred to the first time `forward` is called and `in_units`
        will be inferred from the shape of input data.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
        See document of `Block`.


    Inputs:
        - **data**: if `flatten` is True, `data` should be a tensor with shape
          `(batch_size, x1, x2, ..., xn)`, where x1 * x2 * ... * xn is equal to
          `in_units`. If `flatten` is False, `data` should have shape
          `(x1, x2, ..., xn, in_units)`.

    Outputs:
        - **out**: if `flatten` is True, `out` will be a tensor with shape
          `(batch_size, units)`. If `flatten` is False, `out` will have shape
          `(x1, x2, ..., xn, units)`.

    """

    def __init__(self, units, activation=None, use_bias=True, flatten=True,
                 weight_initializer=None, bias_initializer='zeros', in_units=0,
                 **kwargs):
        super(NormalizedDense, self).__init__(**kwargs)
        self._flatten = flatten
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=(units, ),
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None
            if activation is not None:
                self.act = Activation(activation, prefix=activation + '_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight, bias=None):  # pylint: disable=arguments-differ
        x = F.L2Normalization(x)
        weight = F.L2Normalization(weight)

        act = F.FullyConnected(x, weight, bias, no_bias=bias is None,
                               num_hidden=self._units, flatten=self._flatten,
                               name='fwd')
        if self.act is not None:
            act = self.act(act)
        return act

    def __repr__(self):
        s = '{name}({layout}, {act})'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__, act=self.act
                        if self.act else 'linear', layout='{0} -> {1}'.format(
                            shape[1] if shape[1] else None, shape[0]))


class ThreeCosMul(HybridBlock):
    """The 3CosMul analogy function.

    The 3CosMul analogy function is defined as
    .. math::
        \\arg\\max_{b^* ∈ V}\\frac{\\cos(b^∗, b) \\cos(b^*, a)}{cos(b^*, a^*) + ε}

    """

    def __init__(self, vocab_size, embedding_params=None, k=1, epsilon=0.001,
                 **kwargs):
        super(ThreeCosMul, self).__init__(**kwargs)

        self.epsilon = epsilon
        self.k = k
        with self.name_scope():
            self.embedding_dense = NormalizedDense(vocab_size, flatten=False,
                                                   params=embedding_params)

    def hybrid_forward(self, F, embeddings_words1, embeddings_words2,
                       embeddings_words3):  # pylint: disable=arguments-differ
        """Implement forward computation."""
        all_embeddings = F.concat(embeddings_words1, embeddings_words2,
                                  embeddings_words3, dim=0)
        similarities = self.embedding_dense(all_embeddings)

        # Map cosine similarities to [0, 1]
        similarities = (similarities + 1) / 2

        sim_w1w4, sim_w2w4, sim_w3w4 = F.split(similarities, num_outputs=3,
                                               axis=0)

        pred_idxs = F.topk((sim_w2w4 * sim_w3w4) / (sim_w1w4 + self.epsilon),
                           k=self.k)
        return pred_idxs


class WordEmbeddingAnalogy(HybridBlock):
    """Word embeddings analogy task evaluator.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_size : int
        Dimensionality of the embeddings.
    analogy_function : mxnet.gluon.Block
        Given three mx.nd.NDArray's of word embeddings predict k analogies.
        Vocab_size, embed_size, embedding_params and k are passed in the
        constructor
    embeddings_params : mx.gluon.ParameterDict (optional)
        Parameters of the internal WordEmbeddingSimilarity.embedding Block.

    """

    def __init__(self, vocab_size, embed_size, analogy_function=ThreeCosMul,
                 k=1, exclude_inputs=True, embedding_params=None, **kwargs):
        assert k >= 1

        super(WordEmbeddingAnalogy, self).__init__(**kwargs)

        self.k = k
        self.exclude_inputs = exclude_inputs

        self._internal_k = self.k + 3 * self.exclude_inputs

        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size, embed_size,
                                          weight_initializer=init.Uniform(0.1),
                                          params=embedding_params)
            self.analogy = analogy_function(
                vocab_size=vocab_size, k=self._internal_k,
                embedding_params=self.embedding.params)

    def forward(self, words1, words2, words3):  # pylint: disable=arguments-differ
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
        embeddings_words3 = self.embedding(words3)

        pred_idxs = self.analogy(embeddings_words1, embeddings_words2,
                                 embeddings_words3)

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
