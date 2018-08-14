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


def remove_accidental_hits(candidates, true_samples):
    """Compute a candidates_mask surpressing accidental hits.

    Accidental hits are candidates that occur in the same batch dimension of
    true_samples.

    """
    candidates_np = candidates.asnumpy()
    true_samples_np = true_samples.asnumpy()

    candidates_mask = np.ones(candidates.shape, dtype=np.bool_)
    for j in range(true_samples.shape[1]):
        candidates_mask &= ~(candidates_np == true_samples_np[:, j:j + 1])

    return candidates, mx.nd.array(candidates_mask, ctx=candidates.context)
