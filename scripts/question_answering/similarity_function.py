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

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, Parameter
from mxnet import nd

from .utils import get_combined_dim, combine_tensors

class SimilarityFunction(gluon.HybridBlock):
    """
    A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
    function on the vectors in the last dimension.  For example, the tensors might both have shape
    `(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
    vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
    tensor of shape `(batch_size, sentence_length)`.

    The similarity function could be as simple as a dot product, or it could be a more complex,
    parameterized function.
    """
    default_implementation = 'dot_product'

    def hybrid_forward(self, F, array_1, array_2):
        # pylint: disable=arguments-differ
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.
        """
        raise NotImplementedError


class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.

    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``F.sqrt(ndarray.shape[-1])``, to reduce the
        variance in the result.
    """
    def __init__(self, scale_output=False, **kwargs):
        super(DotProductSimilarity, self).__init__(**kwargs)
        self._scale_output = scale_output

    def hybrid_forward(self, F, array_1, array_2):
        result = (array_1 * array_2).sum(axis=-1)
        if self._scale_output:
            result *= F.sqrt(array_1.shape[-1])
        return result


class CosineSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the cosine similarity between each pair of vectors.  It has
    no parameters.
    """

    def hybrid_forward(self, F, array_1, array_2):
        normalized_array_1 = array_1 / F.norm(array_1, axis=-1, keepdims=True)
        normalized_array_2 = array_2 / F.norm(array_2, axis=-1, keepdims=True)
        return (normalized_array_1 * normalized_array_2).sum(axis=-1)


class BilinearSimilarity(SimilarityFunction):
    """
    This similarity function performs a bilinear transformation of the two input vectors.  This
    function has a matrix of weights ``W`` and a bias ``b``, and the similarity between two vectors
    ``x`` and ``y`` is computed as ``x^T W y + b``.

    Parameters
    ----------
    array_1_dim : ``int``
        The dimension of the first ndarray, ``x``, described above.  This is ``x.shape[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build the weight matrix correctly.
    array_2_dim : ``int``
        The dimension of the second ndarray, ``y``, described above.  This is ``y.shape[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build the weight matrix correctly.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``x^T W y + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 array_1_dim,
                 array_2_dim,
                 activation='linear',
                 **kwargs):
        super(BilinearSimilarity, self).__init__(**kwargs)
        self._weight_matrix = Parameter(shape=(array_1_dim, array_2_dim), init=mx.init.Xavier())
        self._bias = Parameter(shape=(array_1_dim,), init=mx.init.Zero())
        if activation == 'linear':
            self._activation = None
        else:
            self._activation = nn.Activation(activation)

    def hybrid_forward(self, F, array_1, array_2):
        intermediate = F.broadcast_mull(array_1, self._weight_matrix)
        result = F.broadcast_mull(intermediate, array_2).sum(axis=-1)
        if self._activation == None:
            return result
        return self._activation(result + self._bias)

class LinearSimilarity(SimilarityFunction):
    """
    This similarity function performs a dot product between a vector of weights and some
    combination of the two input vectors, followed by an (optional) activation function.  The
    combination used is configurable.

    If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations: ``x``,
    ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give ``x,y,x*y`` as the ``combination`` parameter to this class.  The computed similarity
    function would then be ``w^T [x; y; x*y] + b``, where ``w`` is a vector of weights, ``b`` is a
    bias parameter, and ``[;]`` is vector concatenation.

    Parameters
    ----------
    array_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    array_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : ``str``, optional (default="x,y")
        Described above.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 array_1_dim,
                 array_2_dim,
                 combination='x,y',
                 activation='linear',
                 **kwargs):
        super(LinearSimilarity, self).__init__(**kwargs)
        self._combination = combination
        combined_dim = get_combined_dim(combination, [array_1_dim, array_2_dim])
        self._weight_matrix = Parameter(shape=(array_1_dim, array_2_dim), init=mx.init.Uniform())
        self._bias = Parameter(shape=(array_1_dim,), init=mx.init.Zero())
        if activation == 'linear':
            self._activation = None
        else:
            self._activation = nn.Activation(activation)

    def hybrid_forward(self, F, array_1, array_2):
        combined_tensors = combine_tensors(self._combination, [array_1, array_1])
        dot_product = F.broadcast_mull(combined_tensors, self._weight_vector)
        if self._activation == None:
            return dot_product
        return self._activation(dot_product + self._bias)
