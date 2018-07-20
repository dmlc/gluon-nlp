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
"""Candidate samplers"""

__all__ = ['remove_accidental_hits']

import mxnet as mx
import numpy as np

try:
    from numba import njit
    numba_njit = njit(nogil=True)
except ImportError:

    def numba_njit(func):
        return func


@numba_njit
def _candidates_mask(negatives, true_samples, true_samples_mask):
    assert len(negatives.shape) == 2
    assert len(true_samples.shape) == 2
    assert negatives.shape[0] == true_samples.shape[0]

    negatives_mask = np.ones(negatives.shape)

    for i in range(negatives.shape[0]):
        for j in range(negatives.shape[1]):
            for z in range(true_samples.shape[1]):
                if (true_samples_mask[i, z]
                        and negatives[i, j] == true_samples[i, z]):
                    negatives_mask[i, j] = 0
    return negatives_mask


def remove_accidental_hits(candidates, true_samples, true_samples_mask=None):
    """Compute a candidates_mask surpressing accidental hits.

    Accidental hits are candidates that occur in the same batch dimension of
    true_samples. If true_samples_mask is specified, the masked entries of
    true_samples are ignored when computing the candidates mask.

    """
    candidates_np = candidates.asnumpy()
    true_samples_np = true_samples.asnumpy()
    if true_samples_mask is not None:
        true_samples_mask_np = true_samples_mask.asnumpy()
    else:
        true_samples_mask_np = np.ones_like(true_samples_np)

    candidates_mask = mx.nd.array(
        _candidates_mask(candidates_np, true_samples_np, true_samples_mask_np))

    return candidates, candidates_mask.as_in_context(candidates.context)
