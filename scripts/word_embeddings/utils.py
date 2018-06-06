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
"""Word Embeddings Training Utilities
=====================================

"""

import mxnet as mx


def clip_embeddings_gradients(parameters, max_l2):
    """Clip the gradient norms of embeddings group wise.

    Such that for each group (ie. word or subword) the gradient norm is smaller
    than max_l2.

    Make sure to call after trainer.allreduce_grads() when using multiple
    devices.

    """
    for p in parameters:
        for ctx in p.list_ctx():
            grad = p.grad(ctx)

            # Normalization for row_sparse needs support in mxnet backend
            assert grad.stype == 'default'

            norm = mx.nd.norm(grad, axis=1, keepdims=True)
            scale = mx.nd.divide(max_l2, norm)
            scale = mx.nd.minimum(scale, 1)
            grad[:] = mx.nd.multiply(grad, scale)
