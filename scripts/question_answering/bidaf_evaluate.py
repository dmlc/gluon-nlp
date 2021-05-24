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

"""Performance evaluator - a proxy class used for plugging in official validation script"""
import mxnet as mx
from mxnet import nd, gluon, cpu
from tqdm import tqdm

try:
    from official_squad_eval_script import evaluate
except ImportError:
    from .official_squad_eval_script import evaluate


class PerformanceEvaluator:
    """Plugin to run prediction and performance evaluation via official eval script"""

    def __init__(self, dev_dataloader, dev_dataset, dev_json):
        self._dev_dataloader = dev_dataloader
        self._dev_dataset = dev_dataset
        self._dev_json = dev_json

    def evaluate_performance(self, net, ctx, options):
        """Get results of evaluation by official evaluation script

        Parameters
        ----------
        net : `Block`
            Network
        ctx : `Context`
            Execution context
        options : `Namespace`
            Training arguments

        Returns
        -------
        data : `dict`
            Returns a dictionary of {'exact_match': <value>, 'f1': <value>}
        """

        pred = {}

        # Allows to ensure that start index is always <= than end index
        for _ in ctx:
            answer_mask_matrix = nd.zeros(shape=(1, options.ctx_max_len, options.ctx_max_len),
                                          ctx=cpu(0))
            for idx in range(options.answer_max_len):
                answer_mask_matrix += nd.eye(N=options.ctx_max_len, M=options.ctx_max_len,
                                             k=idx, ctx=cpu(0))

        for idxs, context, query, context_char, query_char, _, _ in tqdm(
                self._dev_dataloader):

            # This is required for multigpu setting. When number of items in the batch is less
            # then number of devices in the context, then it will throw an error if not added
            # fake items
            idxs = PerformanceEvaluator._extend_to_batch_size(options.batch_size * len(ctx),
                                                              idxs, -1)
            query = PerformanceEvaluator._extend_to_batch_size(options.batch_size * len(ctx),
                                                               query)
            context = PerformanceEvaluator._extend_to_batch_size(options.batch_size * len(ctx),
                                                                 context)
            query_char = PerformanceEvaluator._extend_to_batch_size(options.batch_size * len(ctx),
                                                                    query_char)
            context_char = PerformanceEvaluator._extend_to_batch_size(options.batch_size * len(ctx),
                                                                      context_char)

            record_index = gluon.utils.split_and_load(idxs, ctx, even_split=False)
            ctx_words = gluon.utils.split_and_load(context, ctx, even_split=False)
            q_words = gluon.utils.split_and_load(query, ctx, even_split=False)
            q_chars = gluon.utils.split_and_load(query_char, ctx, even_split=False)
            ctx_chars = gluon.utils.split_and_load(context_char, ctx, even_split=False)

            outs = []

            for ri, qw, cw, qc, cc in zip(record_index, q_words, ctx_words,
                                          q_chars, ctx_chars):
                ctx_embedding_state = net.ctx_embedding._contextual_embedding.begin_state(
                    batch_size=ri.shape[0], func=mx.ndarray.zeros, ctx=qw.context)

                modeling_layer_state = net.modeling_layer.begin_state(
                    batch_size=ri.shape[0], func=mx.ndarray.zeros, ctx=qw.context)

                end_index_states = net.output_layer._end_index_lstm.begin_state(
                    batch_size=ri.shape[0], func=mx.ndarray.zeros, ctx=qw.context)

                begin, end = net(qw, cw, qc, cc,
                                 ctx_embedding_state,
                                 modeling_layer_state,
                                 end_index_states)

                outs.append((ri.as_in_context(cpu(0)),
                             begin.as_in_context(cpu(0)),
                             end.as_in_context(cpu(0))))

            for out in outs:
                ri = out[0]
                start = out[1].softmax(axis=1)
                end = out[2].softmax(axis=1)
                start_end_span = PerformanceEvaluator._get_indices(start, end, answer_mask_matrix)

                # iterate over batches
                for idx, start_end in zip(ri, start_end_span):
                    if idx == -1:
                        continue

                    idx = int(idx.asscalar())
                    start = int(start_end[0].asscalar())
                    end = int(start_end[1].asscalar())

                    question_id = self._dev_dataset.get_q_id_by_rec_idx(idx)
                    pred[question_id] = (start, end, self.get_text_result(idx, (start, end)))

        if options.save_prediction_path:
            with open(options.save_prediction_path, 'w') as f:
                for item in pred.items():
                    f.write('{}: {}-{} Answer: {}\n'.format(item[0], item[1][0],
                                                            item[1][1], item[1][2]))

        return evaluate(self._dev_json['data'], {k: v[2] for k, v in pred.items()})

    def get_text_result(self, idx, answer_span):
        """Converts answer span into actual text from paragraph

        Parameters
        ----------
        idx : `int`
            Question index
        answer_span : `Tuple`
            Answer span (start_index, end_index)

        Returns
        -------
        text : `str`
            A chunk of text for provided answer_span or None if answer span cannot be provided
        """

        start, end = answer_span
        context = self._dev_dataset.get_record_by_idx(idx)[8]
        spans = self._dev_dataset.get_record_by_idx(idx)[9]

        # get text from cutting string from the initial context
        # because tokens are hard to combine together
        text = context[spans[start][0]:spans[end][1]]
        return text

    @staticmethod
    def _get_indices(begin, end, answer_mask_matrix):
        r"""Select the begin and end position of answer span.

            At inference time, the predicted span (s, e) is chosen such that
            begin_hat[s] * end_hat[e] is maximized and s ≤ e.

        Parameters
        ----------
        begin : NDArray
            input tensor with shape `(batch_size, context_sequence_length)`
        end : NDArray
            input tensor with shape `(batch_size, context_sequence_length)`

        Returns
        -------
        prediction: Tuple
            Tuple containing first and last token indices of the answer
        """
        begin_hat = begin.reshape(begin.shape + (1,))
        end_hat = end.reshape(end.shape + (1,))
        end_hat = end_hat.transpose(axes=(0, 2, 1))

        result = nd.batch_dot(begin_hat, end_hat) * answer_mask_matrix.slice(
            begin=(0, 0, 0), end=(1, begin_hat.shape[1], begin_hat.shape[1]))
        yp1 = result.max(axis=2).argmax(axis=1, keepdims=True).astype('int32')
        yp2 = result.max(axis=1).argmax(axis=1, keepdims=True).astype('int32')
        return nd.concat(yp1, yp2, dim=-1)

    @staticmethod
    def _extend_to_batch_size(batch_size, prototype, fill_value=0):
        """Provides NDArray, which consist of prototype NDArray and NDArray filled with fill_value
        to batch_size number of items. New NDArray appended to batch dimension (dim=0).

        Parameters
        ----------
        batch_size: ``int``
            Expected value for batch_size dimension (dim=0).
        prototype: ``NDArray``
            NDArray to be extended of shape (batch_size, ...)
        fill_value: ``float``
            Value to use for filling
        """
        if batch_size == prototype.shape[0]:
            return prototype

        new_shape = (batch_size - prototype.shape[0],) + prototype.shape[1:]
        dummy_elements = nd.full(val=fill_value, shape=new_shape, dtype=prototype.dtype,
                                 ctx=prototype.context)

        return nd.concat(prototype, dummy_elements, dim=0)
