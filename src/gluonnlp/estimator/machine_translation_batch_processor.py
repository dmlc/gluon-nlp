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
from ..model.transformer import ParallelTransformer
from ..utils.parallel import Parallel

__all__ = ['MTTransformerBatchProcessor', 'MTGNMTBatchProcessor']

class MTTransformerBatchProcessor(BatchProcessor):
    def __init__(self, rescale_loss=100,
                 batch_size=1024,
                 label_smoothing=None,
                 loss_function=None):
        self.rescale_loss = rescale_loss
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.loss_function = loss_function
        self.parallel_model = None

    def _get_parallel_model(self, estimator):
        if self.label_smoothing is None or self.loss_function is None:
            raise ValueError('label smoothing or loss function cannot be none.')
        if self.parallel_model is None:
            self.parallel_model = ParallelTransformer(estimator.net, self.label_smoothing,
                                                      self.loss_function, self.rescale_loss)
            self.parallel_model = Parallel(len(estimator.context), self.parallel_model)

    def fit_batch(self, estimator, train_batch, batch_axis=0):
        self._get_parallel_model(estimator)
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
            self.parallel_model.put((seq, self.batch_size))
        Ls = [self.parallel_model.get() for _ in range(len(estimator.context))]
        Ls = [l * self.batch_size * self.rescale_loss for l in Ls]
        return data, [target, tgt_word_count - bs], None, Ls

    def evaluate_batch(self, estimator, val_batch, batch_axis=0):
        ctx = estimator.context[0]
        src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids = val_batch
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)

        out, _ = estimator.val_net(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = estimator.val_loss(out, tgt_seq[:, 1:], tgt_valid_length - 1).sum().asscalar()
        inst_ids = inst_ids.asnumpy().astype(np.int32).tolist()
        loss = loss * (tgt_seq.shape[1] - 1)
        val_tgt_valid_length = (tgt_valid_length - 1).sum().asscalar()
        return src_seq, [tgt_seq, val_tgt_valid_length], out, loss

class MTGNMTBatchProcessor(BatchProcessor):
    def __init__(self):
        pass

    def fit_batch(self, estimator, train_batch, batch_axis=0):
        ctx = estimator.context[0]
        src_seq, tgt_seq, src_valid_length, tgt_valid_length = train_batch
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
        with mx.autograd.record():
            out, _ = estimator.net(src_seq, tgt_seq[:, :-1], src_valid_length,
                                   tgt_valid_length - 1)
            loss = estimator.loss(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean()
            loss = loss * (tgt_seq.shape[1] - 1)
            log_loss = loss * tgt_seq.shape[0]
            loss = loss / (tgt_valid_length - 1).mean()
            loss.backward()
        return src_seq, [tgt_seq, (tgt_valid_length - 1).sum()], out, log_loss

    def evaluate_batch(self, estimator, val_batch, batch_axis=0):
        ctx = estimator.context[0]
        src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids = val_batch
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
        out, _ = estimator.val_net(src_seq, tgt_seq[:, :-1], src_valid_length,
                                    tgt_valid_length - 1)
        loss = estimator.val_loss(out, tgt_seq[:, 1:],
                                         tgt_valid_length - 1).sum().asscalar()
        loss = loss * (tgt_seq.shape[1] - 1)
        val_tgt_valid_length = (tgt_valid_length - 1).sum().asscalar()
        return src_seq, [tgt_seq, val_tgt_valid_length], out, loss