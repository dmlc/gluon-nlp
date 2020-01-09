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
# pylint: disable=eval-used, redefined-outer-name
""" Gluon Machine Translation Batch Processor """

import numpy as np
import mxnet as mx
from mxnet.gluon.contrib.estimator import BatchProcessor
from mxnet.gluon.utils import split_and_load

__all__ = ['MTTransformerBatchProcessor']

class MTTransformerBatchProcessor(BatchProcessor):
    def __init__(self, rescale_loss=100, batch_size=1024):
        self.rescale_loss = rescale_loss
        self.batch_size = batch_size

    def fit_batch(self, estimator, train_batch, batch_axis=0):
        data = [shard[0] for shard in train_batch]
        target = [shard[1] for shard in train_batch]
        src_word_count, tgt_word_count, bs = np.sum([(shard[2].sum(),
                                                      shard[3].sum(), shard[0].shape[0]) for shard in train_batch],
                                                    axis=0)
        estimator.tgt_valid_length = tgt_word_count.asscalar() - bs
        seqs = [[seq.as_in_context(context) for seq in shard]
                for context, shard in zip(estimator.context, train_batch)]
        Ls = []
        for seq in seqs:
            estimator.net.put((seq, self.batch_size))
        Ls = [self.estimator.get() for _ in range(len(estimator.context))]
        return data, target, None, Ls

    def evaluate_batch(self, estimator, val_batch, batch_axis=0):
        ctx = estimator.context[0]
        src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids = val_batch
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)

        out, _ = self.eval_net(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = self.evaluation_loss(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        inst_ids = inst_ids.asnumpy().astype(np.int32).tolist()
        loss *= (tgt_seq.shape[1] - 1)
        estimator.val_tgt_valid_length = tgt_seq.shape[1] - 1
        return src_seq, tgt_seq, out, loss
