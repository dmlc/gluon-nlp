import numpy as np
from numpy.testing import assert_allclose
import mxnet as mx
from gluonnlp.data import batchify

import pytest


def test_pad():
    padded = batchify.Pad(pad_val=-1)([mx.nd.array([]), mx.nd.arange(1)]).asnumpy().flatten().tolist()
    assert padded == [-1.0, 0.0]


@pytest.mark.parametrize('odtype', [np.uint8, np.int32, np.int64,
                                    np.float16, np.float32, np.float64])
@pytest.mark.parametrize('idtype', [np.uint8, np.int32, np.int64,
                                    np.float16, np.float32, np.float64])
@pytest.mark.parametrize('pass_dtype', [False, True])
def test_stack_batchify(odtype, idtype, pass_dtype):
    dat = [np.random.randint(5, size=(10,)).astype(idtype) for _ in range(10)]
    batchify_fn = batchify.Stack(dtype=odtype if pass_dtype else None)
    batchify_out = batchify_fn(dat).asnumpy()
    npy_out = np.array(dat)
    assert_allclose(batchify_out, npy_out)
    assert batchify_out.dtype == npy_out.dtype if not pass_dtype else odtype


def test_pad_wrap_batchify():
    def _verify_padded_arr(padded_arr, original_arr, pad_axis, pad_val, pad_length, dtype):
        ndim = original_arr.ndim
        slices_data = [slice(None) for _ in range(ndim)]
        slices_data[pad_axis] = slice(original_arr.shape[axis])
        assert_allclose(padded_arr[tuple(slices_data)], original_arr)
        if original_arr.shape[pad_axis] < pad_length:
            slices_pad_val = [slice(None) for _ in range(ndim)]
            slices_pad_val[axis] = slice(original_arr.shape[pad_axis], None)
            pad_val_in_arr = padded_arr[tuple(slices_pad_val)]
            assert_allclose(pad_val_in_arr, (np.ones_like(pad_val_in_arr) * pad_val).astype(dtype))
    batch_size = 6
    for ndim in range(1, 3):
        for axis in range(-ndim, ndim):
            for length_min, length_max in [(3, 4), (3, 7)]:
                for pad_val in [-1, 0]:
                    for dtype in [np.uint8, np.int32, np.int64, np.float16, np.float32, np.float64]:
                        # Each instance contains a single array
                        for _dtype in [None, dtype]:
                            shapes = [[2 for _ in range(ndim)] for _ in range(batch_size)]
                            for i in range(len(shapes)):
                                shapes[i][axis] = np.random.randint(length_min, length_max)
                            random_data_npy = [np.random.normal(0, 1, shape).astype(dtype)
                                               for shape in shapes]
                            batchify_fn = batchify.Pad(axis=axis, pad_val=pad_val, ret_length=True, dtype=_dtype)
                            batch_data, valid_length = batchify_fn(random_data_npy)
                            batch_data_use_mx, valid_length_use_mx = batchify_fn(
                                [mx.nd.array(ele, dtype=dtype) for ele in random_data_npy])
                            assert_allclose(batch_data_use_mx.asnumpy(), batch_data.asnumpy())
                            assert_allclose(valid_length_use_mx.asnumpy(), valid_length.asnumpy())
                            assert batch_data.dtype == batch_data_use_mx.dtype == dtype
                            assert valid_length.dtype == valid_length_use_mx.dtype == np.int32
                            valid_length = valid_length.asnumpy()
                            batch_data = batch_data.asnumpy()
                            for i in range(batch_size):
                                assert (valid_length[i] == shapes[i][axis])
                                pad_length = max(shape[axis] for shape in shapes)
                                _verify_padded_arr(batch_data[i], random_data_npy[i], axis, pad_val, pad_length, dtype)
                            # Each instance contains 3 arrays, we pad part of them according to index
                            TOTAL_ELE_NUM = 3
                            for pad_index in [[0], [1], [2], [0, 1], [1, 2], [0, 1, 2]]:
                                shapes = [[[2 for _ in range(ndim)] for _ in range(batch_size)]
                                          for _ in range(TOTAL_ELE_NUM)]
                                for j in pad_index:
                                    for i in range(batch_size):
                                        shapes[j][i][axis] = np.random.randint(length_min, length_max)
                                random_data_npy = [tuple(np.random.normal(0, 1, shapes[j][i]).astype(dtype)
                                                         for j in range(TOTAL_ELE_NUM)) for i in range(batch_size)]
                                batchify_fn = []
                                for j in range(TOTAL_ELE_NUM):
                                    if j in pad_index:
                                        batchify_fn.append(batchify.Pad(axis=axis, pad_val=pad_val, ret_length=True,
                                                                        dtype=_dtype))
                                    else:
                                        batchify_fn.append(batchify.Stack(dtype=_dtype))
                                batchify_fn = batchify.Tuple(batchify_fn)
                                ret_use_npy = batchify_fn(random_data_npy)
                                ret_use_mx = batchify_fn(
                                    [tuple(mx.nd.array(ele[i], dtype=dtype) for i in range(TOTAL_ELE_NUM)) for ele in
                                     random_data_npy])
                                for i in range(TOTAL_ELE_NUM):
                                    if i in pad_index:
                                        assert ret_use_npy[i][0].dtype == ret_use_mx[i][0].dtype == dtype
                                        assert ret_use_npy[i][1].dtype == ret_use_mx[i][1].dtype == np.int32
                                        assert_allclose(ret_use_npy[i][0].asnumpy(),
                                                        ret_use_mx[i][0].asnumpy())
                                        assert_allclose(ret_use_npy[i][1].asnumpy(),
                                                        ret_use_mx[i][1].asnumpy())
                                        assert (ret_use_npy[i][1].shape == (batch_size,))
                                    else:
                                        assert ret_use_npy[i].dtype == ret_use_mx[i].dtype == dtype
                                        assert_allclose(ret_use_npy[i].asnumpy(), ret_use_mx[i].asnumpy())
                                for i in range(batch_size):
                                    for j in range(TOTAL_ELE_NUM):
                                        if j in pad_index:
                                            batch_data, valid_length = ret_use_npy[j][0].asnumpy(), \
                                                                       ret_use_npy[j][1].asnumpy()
                                            assert (valid_length[i] == shapes[j][i][axis])
                                        else:
                                            batch_data = ret_use_npy[j].asnumpy()
                                        pad_length = max(ele[j].shape[axis] for ele in random_data_npy)
                                        _verify_padded_arr(batch_data[i], random_data_npy[i][j], axis, pad_val,
                                                           pad_length,
                                                           dtype)
                        for _dtype in [np.float16, np.float32]:
                            shapes = [[2 for _ in range(ndim)] for _ in range(batch_size)]
                            for i in range(len(shapes)):
                                shapes[i][axis] = np.random.randint(length_min, length_max)
                            random_data_npy = [np.random.normal(0, 1, shape).astype(dtype)
                                               for shape in shapes]
                            batchify_fn = batchify.Pad(axis=axis, pad_val=pad_val, ret_length=True, dtype=_dtype)
                            batch_data, valid_length = batchify_fn(random_data_npy)
                            batch_data_use_mx, valid_length_use_mx = batchify_fn(
                                [mx.nd.array(ele, dtype=dtype) for ele in random_data_npy])
                            assert_allclose(valid_length_use_mx.asnumpy(), valid_length.asnumpy())
                            assert batch_data.dtype == batch_data_use_mx.dtype == _dtype
                            assert valid_length.dtype == valid_length_use_mx.dtype == np.int32

