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
"""Batchify functions. They can be used in Gluon data loader to help combine individual samples
into batches for fast processing."""
__all__ = ['Stack', 'Pad', 'Tuple']

import logging

import numpy as np
import mxnet as mx


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, use_shared_mem, dtype):
    """Inner Implementation of the Pad batchify

    Parameters
    ----------
    arrs : list
    pad_axis : int
    pad_val : number
    use_shared_mem : bool, default False

    Returns
    -------
    ret : NDArray
    original_length : NDArray
    """
    if isinstance(arrs[0], mx.nd.NDArray):
        dtype = arrs[0].dtype if dtype is None else dtype
        arrs = [arr.asnumpy() for arr in arrs]
    elif not isinstance(arrs[0], np.ndarray):
        arrs = [np.asarray(ele) for ele in arrs]
    else:
        dtype = arrs[0].dtype if dtype is None else dtype

    original_length = [ele.shape[pad_axis] for ele in arrs]
    max_size = max(original_length)

    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs), ) + tuple(ret_shape)

    ret = np.full(shape=ret_shape, fill_value=pad_val, dtype=dtype)

    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[i] = arr
        else:
            slices = [slice(None) for _ in range(arr.ndim)]
            slices[pad_axis] = slice(0, arr.shape[pad_axis])
            if slices[pad_axis].start != slices[pad_axis].stop:
                slices = [slice(i, i + 1)] + slices
                ret[tuple(slices)] = arr

    ctx = mx.Context('cpu_shared', 0) if use_shared_mem else mx.cpu()
    ret = mx.nd.array(ret, ctx=ctx, dtype=dtype)
    original_length = mx.nd.array(original_length, ctx=ctx, dtype=np.int32)

    return ret, original_length


def _stack_arrs(arrs, use_shared_mem, dtype):
    if isinstance(arrs[0], mx.nd.NDArray):
        dtype = arrs[0].dtype if dtype is None else dtype
        if use_shared_mem:
            out = mx.nd.empty((len(arrs),) + arrs[0].shape, dtype=dtype,
                              ctx=mx.Context('cpu_shared', 0))
            return mx.nd.stack(*arrs, out=out)
        else:
            return mx.nd.stack(*arrs)
    else:
        out = np.asarray(arrs)
        dtype = out.dtype if dtype is None else dtype
        if use_shared_mem:
            return mx.nd.array(out, ctx=mx.Context('cpu_shared', 0), dtype=dtype)
        else:
            return mx.nd.array(out, dtype=dtype)


class Stack(object):
    r"""Stack the input data samples to construct the batch.

    The N input samples must have the same shape/length and will be stacked to construct a batch.

    Parameters
    ----------
    dtype : str or numpy.dtype, default None
        The value type of the output. If it is set to None, the input data type is used.

    Examples
    --------
    >>> from gluonnlp.data import batchify
    >>> # Stack multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6, 8]
    >>> c = [8, 9, 1, 2]
    >>> gluonnlp.data.batchify.Stack()([a, b, c])
    <BLANKLINE>
    [[1 2 3 4]
     [4 5 6 8]
     [8 9 1 2]]
    <NDArray 3x4 @cpu_shared(0)>
    >>> # Stack multiple numpy.ndarrays
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> gluonnlp.data.batchify.Stack()([a, b])
    <BLANKLINE>
    [[[1 2 3 4]
      [5 6 7 8]]
    <BLANKLINE>
     [[5 6 7 8]
      [1 2 3 4]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    >>> # Stack multiple NDArrays
    >>> a = mx.nd.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = mx.nd.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> gluonnlp.data.batchify.Stack()([a, b])
    <BLANKLINE>
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
    <BLANKLINE>
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    """
    def __init__(self, dtype=None):
        self._dtype = dtype

    def __call__(self, data):
        """Batchify the input data

        Parameters
        ----------
        data : list
            The input data samples

        Returns
        -------
        batch_data : NDArray
        """
        return _stack_arrs(data, True, self._dtype)


class Pad(object):
    """Return a callable that pads and stacks data.

    Parameters
    ----------
    axis : int, default 0
        The axis to pad the arrays. The arrays will be padded to the largest dimension at
        `axis`. For example, assume the input arrays have shape
        (10, 8, 5), (6, 8, 5), (3, 8, 5) and the `axis` is 0. Each input will be padded into
        (10, 8, 5) and then stacked to form the final output, which has shape（3, 10, 8, 5).
    pad_val : float or int, default 0
        The padding value.
    ret_length : bool, default False
        Whether to return the valid length in the output.
    dtype : str or numpy.dtype, default None
        The value type of the output. If it is set to None, the input data type is used.

    Examples
    --------
    >>> # Inputs are multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> gluonnlp.data.batchify.Pad()([a, b, c])
    <BLANKLINE>
    [[1. 2. 3. 4.]
     [4. 5. 6. 0.]
     [8. 2. 0. 0.]]
    <NDArray 3x4 @cpu_shared(0)>
    >>> # Also output the lengths
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batch, length = gluonnlp.data.batchify.Pad(ret_length=True)([a, b, c])
    >>> batch
    <BLANKLINE>
    [[1. 2. 3. 4.]
     [4. 5. 6. 0.]
     [8. 2. 0. 0.]]
    <NDArray 3x4 @cpu_shared(0)>
    >>> length
    <BLANKLINE>
    [4 3 2]
    <NDArray 3 @cpu_shared(0)>
    >>> # Inputs are multiple ndarrays
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 8], [1, 2]])
    >>> gluonnlp.data.batchify.Pad(axis=1, pad_val=-1)([a, b])
    <BLANKLINE>
    [[[ 1  2  3  4]
      [ 5  6  7  8]]
    <BLANKLINE>
     [[ 5  8 -1 -1]
      [ 1  2 -1 -1]]]
    <NDArray 2x2x4 @cpu_shared(0)>
    """
    def __init__(self, axis=0, pad_val=0, ret_length=False, dtype=None):
        self._axis = axis
        assert isinstance(axis, int), 'axis must be an integer! ' \
                                      'Received axis=%s, type=%s.' % (str(axis),
                                                                      str(type(axis)))
        self._pad_val = pad_val
        self._ret_length = ret_length
        self._dtype = dtype
        self._warned = False

    def __call__(self, data):
        """Batchify the input data.

        The input can be list of numpy.ndarray, list of numbers or list of
        mxnet.nd.NDArray. Inputting mxnet.nd.NDArray is discouraged as each
        array need to be converted to numpy for efficient padding.

        The arrays will be padded to the largest dimension at `axis` and then
        stacked to form the final output. In addition, the function will output
        the original dimensions at the `axis` if ret_length is turned on.

        Parameters
        ----------
        data : List[np.ndarray] or List[List[dtype]] or List[mx.nd.NDArray]
            List of samples to pad and stack.

        Returns
        -------
        batch_data: NDArray
            Data in the minibatch. Shape is (N, ...)
        valid_length: NDArray, optional
            The sequences' original lengths at the padded axis. Shape is (N,). This will only be
            returned in `ret_length` is True.

        """

        if isinstance(data[0], mx.nd.NDArray) and not self._warned:
            self._warned = True
            logging.warning(
                'Using Pad with NDArrays is discouraged for speed reasons. '
                'Instead you should pad your data while it is still a list '
                'and before converting to an NDArray. '
                'Alternatively you can consider inputting a numpy.ndarray.')
        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr, original_length = _pad_arrs_to_max_length(data, self._axis,
                                                                  self._pad_val, True,
                                                                  self._dtype)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError


class Tuple(object):
    """Wrap multiple batchify functions together. The input functions will be applied
    to the corresponding input fields.

    Each data sample should be a list or tuple containing multiple attributes. The `i`th batchify
    function stored in `Tuple` will be applied on the `i`th attribute. For example, each
    data sample is (nd_data, label). You can wrap two batchify functions using
    `Tuple(DataBatchify, LabelBatchify)` to batchify nd_data and label correspondingly.

    Parameters
    ----------
    fn : list or tuple or callable
        The batchify functions to wrap.
    *args : tuple of callable
        The additional batchify functions to wrap.

    Examples
    --------
    >>> a = ([1, 2, 3, 4], 0)
    >>> b = ([5, 7], 1)
    >>> c = ([1, 2, 3, 4, 5, 6, 7], 0)
    >>> f1, f2 = gluonnlp.data.batchify.Tuple(gluonnlp.data.batchify.Pad(),
    ...                                       gluonnlp.data.batchify.Stack())([a, b])
    >>> f1
    <BLANKLINE>
    [[1. 2. 3. 4.]
     [5. 7. 0. 0.]]
    <NDArray 2x4 @cpu_shared(0)>
    >>> f2
    <BLANKLINE>
    [0 1]
    <NDArray 2 @cpu_shared(0)>

    """
    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn, ) + args
        for i, ele_fn in enumerate(self._fn):
            assert hasattr(ele_fn, '__call__'), 'Batchify functions must be callable! ' \
                                                'type(fn[%d]) = %s' % (i, str(type(ele_fn)))

    def __call__(self, data):
        """Batchify the input data.

        Parameters
        ----------
        data : list
            The samples to batchfy. Each sample should contain N attributes.

        Returns
        -------
        ret : tuple
            A tuple of length N. Contains the batchified result of each attribute in the input.
        """
        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contains' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)
