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

import warnings
import mxnet as mx
from mxnet.test_utils import default_context, assert_almost_equal, rand_ndarray
import numpy as np
from gluonnlp import optimizer

def compare_ndarray_tuple(t1, t2, rtol=None, atol=None):
    """Compare ndarray tuple."""
    if t1 is not None and t2 is not None:
        if isinstance(t1, tuple):
            for s1, s2 in zip(t1, t2):
                compare_ndarray_tuple(s1, s2, rtol, atol)
        else:
            assert_almost_equal(t1.asnumpy(), t2.asnumpy(), rtol=rtol, atol=atol)


def compare_optimizer(opt1, opt2, shape, dtype, w_stype='default', g_stype='default',
                      rtol=1e-4, atol=1e-5, compare_states=True):
    """Compare opt1 and opt2."""
    if w_stype == 'default':
        w2 = mx.random.uniform(shape=shape, ctx=default_context(), dtype=dtype)
        w1 = w2.copyto(default_context())
    elif w_stype == 'row_sparse' or w_stype == 'csr':
        w2 = rand_ndarray(shape, w_stype, density=1, dtype=dtype)
        w1 = w2.copyto(default_context()).tostype('default')
    else:
        raise Exception("type not supported yet")
    if g_stype == 'default':
        g2 = mx.random.uniform(shape=shape, ctx=default_context(), dtype=dtype)
        g1 = g2.copyto(default_context())
    elif g_stype == 'row_sparse' or g_stype == 'csr':
        g2 = rand_ndarray(shape, g_stype, dtype=dtype)
        g1 = g2.copyto(default_context()).tostype('default')
    else:
        raise Exception("type not supported yet")

    state1 = opt1.create_state_multi_precision(0, w1)
    state2 = opt2.create_state_multi_precision(0, w2)
    if compare_states:
        compare_ndarray_tuple(state1, state2)

    opt1.update_multi_precision(0, w1, g1, state1)
    opt2.update_multi_precision(0, w2, g2, state2)
    if compare_states:
        compare_ndarray_tuple(state1, state2, rtol=rtol, atol=atol)
    assert_almost_equal(w1.asnumpy(), w2.asnumpy(), rtol=rtol, atol=atol)

# BERT ADAM
class PyBERTAdam(mx.optimizer.Optimizer):
    """python reference implemenation of BERT style adam"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 wd=0, **kwargs):
        super(PyBERTAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.wd = wd

    def create_state_multi_precision(self, index, weight):
        """multi-precision state creation function."""
        weight_master_copy = None
        if self.multi_precision and weight.dtype == np.float16:
            weight_master_copy = weight.astype(np.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == np.float16 and not self.multi_precision:
            warnings.warn('Accumulating with float16 in optimizer can lead to '
                          'poor accuracy or slow convergence. '
                          'Consider using multi_precision=True option of the '
                          'BERTAdam optimizer')
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
        The weight data
        """
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update_multi_precision(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
        An unique integer key used to index the parameters
        weight : NDArray
        weight ndarray
        grad : NDArray
        grad ndarray
        state : NDArray or other objects returned by init_state
        The auxiliary state used in optimization.
        """
        use_multi_precision = self.multi_precision and weight.dtype == np.float16
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        if use_multi_precision:
            mean, variance = state[0]
            weight32 = state[1]
        else:
            mean, variance = state
            weight32 = weight.copy()
        grad = grad.astype('float32') * self.rescale_grad
        # clip gradients
        if self.clip_gradient is not None:
            mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient, out=grad)
        # update mean
        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        # update variance
        variance[:] = self.beta2 * variance + (1 - self.beta2) * grad.square()
        # include weight decay
        update = mean / (mx.nd.sqrt(variance) + self.epsilon) + wd * weight32
        # update weight
        if use_multi_precision:
            weight32 -= lr * update
            weight[:] = weight32.astype(weight.dtype)
        else:
            weight -= lr * update


def test_bert_adam():
    opt1 = PyBERTAdam
    opt2 = optimizer.BERTAdam
    shape = (3, 4, 5)
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}]
    for dtype in [np.float16, np.float32]:
        for cg_option in cg_options:
            for rg_option in rg_options:
                for wd_option in wd_options:
                    kwarg = {}
                    kwarg.update(cg_option)
                    kwarg.update(rg_option)
                    kwarg.update(wd_option)
                    if np.float16 == dtype:
                        kwarg['multi_precision'] = True
                        rtol = 1e-3
                    else:
                        rtol = 1e-4
                    try:
                        compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype,
                                          rtol=rtol, atol=2e-5)
                    except ImportError:
                        print('skipping test_bert_adam() because an old version of MXNet is found')
                        return
