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

"""Collection of general purpose similarity functions"""

__all__ = ['SimilarityFunction', 'DotProductSimilarity', 'CosineSimilarity',
           'BilinearSimilarity', 'LinearSimilarity']

import mxnet as mx
from mxnet import gluon, initializer
from mxnet.gluon import nn, Parameter


def _get_combination(combination, tensors):
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == '*':
            return first_tensor * second_tensor
        elif operation == '/':
            return first_tensor / second_tensor
        elif operation == '+':
            return first_tensor + second_tensor
        elif operation == '-':
            return first_tensor - second_tensor
        else:
            raise NotImplementedError


def combine_tensors(F, combination, tensors):
    """
    Combines a list of tensors using element-wise operations and concatenation, specified by a
    ``combination`` string.  The string refers to (1-indexed) positions in the input tensor list,
    and looks like ``"1,2,1+2,3-1"``.

    We allow the following kinds of combinations: ``x``, ``x*y``, ``x+y``, ``x-y``, and ``x/y``,
    where ``x`` and ``y`` are positive integers less than or equal to ``len(tensors)``.  Each of
    the binary operations is performed elementwise.  You can give as many combinations as you want
    in the ``combination`` string.  For example, for the input string ``"1,2,1*2"``, the result
    would be ``[1;2;1*2]``, as you would expect, where ``[;]`` is concatenation along the last
    dimension.

    If you have a fixed, known way to combine tensors that you use in a model, you should probably
    just use something like ``ndarray.cat([x_tensor, y_tensor, x_tensor * y_tensor])``.  This
    function adds some complexity that is only necessary if you want the specific combination used
    to be `configurable`.

    If you want to do any element-wise operations, the tensors involved in each element-wise
    operation must have the same shape.

    This function also accepts ``x`` and ``y`` in place of ``1`` and ``2`` in the combination
    string.
    """
    combination = combination.replace('x', '1').replace('y', '2')
    to_concatenate = [_get_combination(piece, tensors) for piece in combination.split(',')]
    return F.concat(*to_concatenate, dim=-1)


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

    def hybrid_forward(self, F, array_1, array_2):  # pylint: disable=arguments-differ
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
    This similarity function simply computes the cosine similarity between each pair of vectors.
    It has no parameters.
    """

    def hybrid_forward(self, F, array_1, array_2):
        normalized_array_1 = F.broadcast_div(array_1, F.norm(array_1, axis=-1, keepdims=True))
        normalized_array_2 = F.broadcast_div(array_2, F.norm(array_2, axis=-1, keepdims=True))
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
        self._weight_matrix = Parameter(name='weight_matrix',
                                        shape=(array_1_dim, array_2_dim), init=mx.init.Xavier())
        self._bias = Parameter(name='bias', shape=(array_1_dim,), init=mx.init.Zero())

        if activation == 'linear':
            self._activation = None
        else:
            self._activation = nn.Activation(activation)

    def hybrid_forward(self, F, array_1, array_2):
        intermediate = F.broadcast_mull(array_1, self._weight_matrix)
        result = F.broadcast_mull(intermediate, array_2).sum(axis=-1)

        if not self._activation:
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
                 use_bias=False,
                 combination='x,y',
                 activation='linear',
                 **kwargs):
        super(LinearSimilarity, self).__init__(**kwargs)
        self.combination = combination
        self.use_bias = use_bias
        self.array_1_dim = array_1_dim
        self.array_2_dim = array_2_dim

        if activation == 'linear':
            self._activation = None
        else:
            self._activation = nn.Activation(activation)

        with self.name_scope():
            self.weight_matrix = self.params.get('weight_matrix',
                                                 shape=(array_2_dim, array_1_dim),
                                                 init=initializer.Uniform())
            if use_bias:
                self.bias = self.params.get('bias',
                                            shape=(array_2_dim,),
                                            init=initializer.Zero())

    def hybrid_forward(self, F, array_1, array_2, weight_matrix, bias=None):
        # pylint: disable=arguments-differ
        combined_tensors = combine_tensors(F, self.combination, [array_1, array_2])
        dot_product = F.FullyConnected(combined_tensors, weight_matrix, bias=bias, flatten=False,
                                       no_bias=not self.use_bias, num_hidden=self.array_2_dim)

        if not self._activation:
            return dot_product

        return self._activation(dot_product + bias)
