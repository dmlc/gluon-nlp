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
"""Log Uniform Candidate Sampler"""

import math
import numpy as np
from mxnet import ndarray, gluon


class LogUniformSampler(gluon.block.Block):
    """Draw random samples from an approximately log-uniform or Zipfian distribution.

    This operation randomly samples *num_sampled* candidates the range of integers [0, range_max).
    The elements of sampled_candidates are drawn without replacement from the base distribution.

    The base distribution for this operator is an approximately log-uniform or Zipfian distribution:

    P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)

    This sampler is useful when the true classes approximately follow such a distribution.

    For example, if the classes represent words in a lexicon sorted in decreasing order of
    frequency. If your classes are not ordered by decreasing frequency, do not use this op.

    Additionally, it also returns the number of times each of the
    true classes and the sampled classes is expected to occur.

    As the candidates are drawn without replacement, the expected count for the sampled candidates
    and true classes are approximated. If the candidates are drawn with `num_tries` draws, we assume
    (falsely) that the number of tries to get a batch of batch_size distinct values is always
    `num_tries`, and the probability that the value is in a batch is 1 - (1-p)**num_tries.

    Parameters
    ----------
    num_sampled: int
        The number of classes to randomly sample.
    range_max: int
        The number of possible classes.
    dtype: str or np.dtype
        The dtype for outputs
    """
    def __init__(self, range_max, num_sampled, dtype=None, **kwargs):
        super(LogUniformSampler, self).__init__(**kwargs)
        self._num_sampled = num_sampled
        self._log_range = math.log(range_max + 1)
        self._dtype = np.float32 if dtype is None else dtype
        self._range_max = range_max

    def _prob_helper(self, num_tries, prob):
        return (num_tries.astype('float64') * (-prob).log1p()).expm1() * -1

    def forward(self, true_classes): # pylint: disable=arguments-differ
        """Draw samples from log uniform distribution and returns sampled candidates,
        expected count for true classes and sampled classes.

        Parameters
        ----------
        true_classes: NDArray
            The true classes.

        Returns
        -------
        samples: NDArray
            The sampled candidate classes.
        expected_count_sample: NDArray
            The expected count for sampled candidates.
        expected_count_true: NDArray
            The expected count for true classes in the same shape as `true_classes`.
        """
        num_sampled = self._num_sampled
        ctx = true_classes.context
        num_tries = 0
        log_range = math.log(self._range_max + 1)

        # sample candidates
        f = ndarray._internal._sample_unique_zipfian
        sampled_classes, num_tries = f(self._range_max, shape=(1, num_sampled))
        sampled_classes = sampled_classes.reshape((-1,))
        sampled_classes = sampled_classes.as_in_context(ctx)
        num_tries = num_tries.as_in_context(ctx)

        # expected count for true classes
        true_cls = true_classes.as_in_context(ctx).astype('float64')
        prob_true = ((true_cls + 2.0) / (true_cls + 1.0)).log() / log_range
        count_true = self._prob_helper(num_tries, prob_true)
        # expected count for sampled classes
        sampled_classes = ndarray.array(sampled_classes, ctx=ctx, dtype='int64')
        sampled_cls_fp64 = sampled_classes.astype('float64')
        prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)).log() / log_range
        count_sampled = self._prob_helper(num_tries, prob_sampled)
        # convert to dtype
        sampled_classes = sampled_classes.astype(self._dtype, copy=False)
        count_true = count_true.astype(self._dtype, copy=False)
        count_sampled = count_sampled.astype(self._dtype, copy=False)
        return sampled_classes, count_sampled, count_true
