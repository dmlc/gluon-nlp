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

from mxnet import nd

def get_combined_dim(combination, tensor_dims):
    """
    For use with :func:`combine_tensors`.  This function computes the resultant dimension when
    calling ``combine_tensors(combination, tensors)``, when the tensor dimension is known.  This is
    necessary for knowing the sizes of weight matrices when building models that use
    ``combine_tensors``.

    Parameters
    ----------
    combination : ``str``
        A comma-separated list of combination pieces, like ``"1,2,1*2"``, specified identically to
        ``combination`` in :func:`combine_tensors`.
    tensor_dims : ``List[int]``
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to :func:`combine_tensors`.
    """
    combination = combination.replace('x', '1').replace('y', '2')
    return sum([_get_combination_dim(piece, tensor_dims) for piece in combination.split(',')])


def _get_combination_dim(combination, tensor_dims):
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        return first_tensor_dim

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

def combine_tensors(combination, tensors):
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
    return nd.cat(to_concatenate, dim=-1)


def masked_softmax(vector, mask):
    """
    ``nd.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = nd.softmax(vector, axis=-1)
    else:
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = nd.softmax(vector * mask, axis=-1)
        result = result * mask
        result = result / (result.sum(axis=1, keepdims=True) + 1e-13)
    return result


def masked_log_softmax(vector, mask):
    """
    ``nd.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.

    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return nd.log_softmax(vector, axis=1)


def _last_dimension_applicator(function_to_apply,
                               tensor,
                               mask):
    """
    Takes a tensor with 3 or more dimensions and applies a function over the last dimension.  We
    assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
    has shape ``(batch_size, sequence_length)``.  We first unsqueeze and expand the mask so that it
    has the same shape as the tensor, then flatten them both to be 2D, pass them through
    the function and put the tensor back in its original shape.
    """
    tensor_shape = tensor.shape
    reshaped_tensor = tensor.reshape(-1, tensor.shape[-1])
    if mask is not None:
        while mask.shape[0] < tensor.shape[0]:
            mask = mask.expand_dims(1)
        mask = mask.broadcast_to(tensor).contiguous().float()
        mask = mask.reshape(-1, mask.shape[-1])
    reshaped_result = function_to_apply(reshaped_tensor, mask)
    return reshaped_result.reshape(*tensor_shape)


def last_dim_softmax(tensor, mask):
    """
    Takes a tensor with 3 or more dimensions and does a masked softmax over the last dimension.  We
    assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
    has shape ``(batch_size, sequence_length)``.
    """
    return _last_dimension_applicator(masked_softmax, tensor, mask)


def last_dim_log_softmax(tensor, mask):
    """
    Takes a tensor with 3 or more dimensions and does a masked log softmax over the last dimension.
    We assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
    has shape ``(batch_size, sequence_length)``.
    """
    return _last_dimension_applicator(masked_log_softmax, tensor, mask)


def weighted_sum(matrix, attention):
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """

    if attention.shape[0] == 2 and matrix.shape[0] == 3:
        return attention.expand_dims(1).batch_dot(matrix).squeeze(1)
    if attention.shape[0] == 3 and matrix.shape[0] == 3:
        return attention.batch_dot(matrix)
    if matrix.shape[0] - 1 < attention.shape[0]:
        expanded_size = list(matrix.shape)
        for i in range(attention.shape[0] - matrix.shape[0] + 1):
            matrix = matrix.expand_dims(1)
            expanded_size.insert(i + 1, attention.shape[i + 1])
        matrix = matrix.broadcast_to(*expanded_size)
    intermediate = attention.expand_dims(-1).broadcast_to(matrix.shape) * matrix
    return intermediate.sum(axis=-2)


def replace_masked_values(tensor, mask, replace_with):
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.
    """
    # We'll build a tensor of the same shape as `tensor`, zero out masked values, then add back in
    # the `replace_with` value.
    one_minus_mask = 1.0 - mask
    values_to_add = replace_with * one_minus_mask
    return tensor * mask + values_to_add