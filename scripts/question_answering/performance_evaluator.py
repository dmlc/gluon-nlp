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
import multiprocessing
from mxnet import nd, gluon, cpu
from mxnet.gluon.data import DataLoader, ArrayDataset

from .data_processing import SQuADTransform
from .official_squad_eval_script import evaluate
from .utils import extend_to_batch_size


class PerformanceEvaluator:
    """Plugin to run prediction and performance evaluation via official eval script"""
    def __init__(self, tokenizer, evaluation_dataset, json_data, question_id_mapper):
        self._tokenizer = tokenizer
        self._evaluation_dataset = evaluation_dataset
        self._json_data = json_data
        self._mapper = question_id_mapper

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

        eval_dataset = ArrayDataset([(self._mapper.question_id_to_idx[r[1]], r[2], r[3], r[4], r[5])
                                     for r in self._evaluation_dataset])
        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=len(ctx) * options.batch_size,
                                     last_batch='keep',
                                     pin_memory=True,
                                     num_workers=(multiprocessing.cpu_count() - len(ctx) - 2))

        for data in eval_dataloader:
            record_index, q_words, ctx_words, q_chars, ctx_chars = data

            record_index = extend_to_batch_size(options.batch_size * len(ctx), record_index, -1)
            q_words = extend_to_batch_size(options.batch_size * len(ctx), q_words)
            ctx_words = extend_to_batch_size(options.batch_size * len(ctx), ctx_words)
            q_chars = extend_to_batch_size(options.batch_size * len(ctx), q_chars)
            ctx_chars = extend_to_batch_size(options.batch_size * len(ctx), ctx_chars)

            record_index = gluon.utils.split_and_load(record_index, ctx, even_split=False)
            q_words = gluon.utils.split_and_load(q_words, ctx, even_split=False)
            ctx_words = gluon.utils.split_and_load(ctx_words, ctx, even_split=False)
            q_chars = gluon.utils.split_and_load(q_chars, ctx, even_split=False)
            ctx_chars = gluon.utils.split_and_load(ctx_chars, ctx, even_split=False)

            outs = []

            for ri, qw, cw, qc, cc in zip(record_index, q_words, ctx_words,
                                          q_chars, ctx_chars):
                begin, end = net(qw, cw, qc, cc)
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
                    idx = int(idx.asscalar())
                    start = int(start_end[0].asscalar())
                    end = int(start_end[1].asscalar())

                    if idx in self._mapper.idx_to_question_id:
                        question_id = self._mapper.idx_to_question_id[idx]
                        pred[question_id] = (start, end, self.get_text_result(idx, (start, end)))

        if options.save_prediction_path:
            with open(options.save_prediction_path, 'w') as f:
                for item in pred.items():
                    f.write('{}: {}-{} Answer: {}\n'.format(item[0], item[1][0],
                                                            item[1][1], item[1][2]))

        return evaluate(self._json_data['data'], {k: v[2] for k, v in pred.items()})

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

        if start > end:
            return ''

        question_id = self._mapper.idx_to_question_id[idx]
        context = self._mapper.question_id_to_context[question_id]
        context_tokens = self._tokenizer(context, lower_case=True)
        indices = SQuADTransform.get_char_indices(context, context_tokens)

        # get text from cutting string from the initial context
        # because tokens are hard to combine together
        text = context[indices[start][0]:indices[end][1]]
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
