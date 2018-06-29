import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
from gluonnlp.data import batchify

import pytest

def test_pad():
    padded = batchify.Pad(pad_val=-1)([mx.nd.array([]), mx.nd.arange(1)]).asnumpy().flatten().tolist()
    assert padded == [-1.0, 0.0]

def test_stack_batchify():
    batchify_fn = batchify.Stack()
    dat = [np.random.randint(5) for _ in range(10)]
    assert_allclose(batchify_fn(dat).asnumpy(), np.array(dat))

def test_pad_wrap_batchify():
    def _verify_padded_arr(padded_arr, original_arr, pad_axis, pad_val, pad_length):
        ndim = original_arr.ndim
        slices_data = [slice(None) for _ in range(ndim)]
        slices_data[pad_axis] = slice(original_arr.shape[axis])
        assert_allclose(padded_arr[tuple(slices_data)], original_arr)
        if original_arr.shape[pad_axis] < pad_length:
            slices_pad_val = [slice(None) for _ in range(ndim)]
            slices_pad_val[axis] = slice(original_arr.shape[pad_axis], None)
            pad_val_in_arr = padded_arr[tuple(slices_pad_val)]
            assert_allclose(pad_val_in_arr, np.ones_like(pad_val_in_arr) * pad_val)
    batch_size = 6
    for ndim in range(1, 3):
        for axis in range(-ndim, ndim):
            for length_min, length_max in [(3, 4), (3, 7)]:
                for pad_val in [-1, 0]:
                    # Each instance contains a single array
                    shapes = [[2 for _ in range(ndim)] for _ in range(batch_size)]
                    for i in range(len(shapes)):
                        shapes[i][axis] = np.random.randint(length_min, length_max)
                    random_data_npy = [np.random.normal(0, 1, shape) for shape in shapes]
                    batchify_fn = batchify.Pad(axis=axis, pad_val=pad_val, ret_length=True)
                    batch_data, valid_length = batchify_fn(random_data_npy)
                    batch_data_use_mx, valid_length_use_mx = batchify_fn([mx.nd.array(ele) for ele in random_data_npy])
                    assert_allclose(batch_data_use_mx.asnumpy(), batch_data.asnumpy())
                    assert_allclose(valid_length_use_mx.asnumpy(), valid_length.asnumpy())
                    valid_length = valid_length.asnumpy()
                    batch_data = batch_data.asnumpy()
                    for i in range(batch_size):
                        assert (valid_length[i] == shapes[i][axis])
                        pad_length = max(shape[axis] for shape in shapes)
                        _verify_padded_arr(batch_data[i], random_data_npy[i], axis, pad_val, pad_length)
                    # Each instance contains 3 arrays, we pad part of them according to index
                    TOTAL_ELE_NUM = 3
                    for pad_index in [[0], [1], [2], [0, 1], [1, 2], [0, 1, 2]]:
                        shapes = [[[2 for _ in range(ndim)] for _ in range(batch_size)]
                                  for _ in range(TOTAL_ELE_NUM)]
                        for j in pad_index:
                            for i in range(batch_size):
                                shapes[j][i][axis] = np.random.randint(length_min, length_max)
                        random_data_npy = [tuple(np.random.normal(0, 1, shapes[j][i])
                                                 for j in range(TOTAL_ELE_NUM)) for i in range(batch_size)]
                        batchify_fn = []
                        for j in range(TOTAL_ELE_NUM):
                            if j in pad_index:
                                batchify_fn.append(batchify.Pad(axis=axis, pad_val=pad_val, ret_length=True))
                            else:
                                batchify_fn.append(batchify.Stack())
                        batchify_fn = batchify.Tuple(batchify_fn)
                        ret_use_npy = batchify_fn(random_data_npy)
                        ret_use_mx = batchify_fn([tuple(mx.nd.array(ele[i]) for i in range(TOTAL_ELE_NUM)) for ele in random_data_npy])
                        for i in range(TOTAL_ELE_NUM):
                            if i in pad_index:
                                assert_allclose(ret_use_npy[i][0].asnumpy(),
                                                ret_use_mx[i][0].asnumpy())
                                assert_allclose(ret_use_npy[i][1].asnumpy(),
                                                ret_use_mx[i][1].asnumpy())
                                assert(ret_use_npy[i][1].shape == (batch_size,))
                            else:
                                assert_allclose(ret_use_npy[i].asnumpy(), ret_use_mx[i].asnumpy())
                        for i in range(batch_size):
                            for j in range(TOTAL_ELE_NUM):
                                if j in pad_index:
                                    batch_data, valid_length = ret_use_npy[j][0].asnumpy(),\
                                                               ret_use_npy[j][1].asnumpy()
                                    assert (valid_length[i] == shapes[j][i][axis])
                                else:
                                    batch_data = ret_use_npy[j].asnumpy()
                                pad_length = max(ele[j].shape[axis] for ele in random_data_npy)
                                _verify_padded_arr(batch_data[i], random_data_npy[i][j], axis, pad_val, pad_length)
