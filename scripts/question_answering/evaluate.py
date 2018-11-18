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

# pylint: disable=
r"""
In the evaluate function, we use the offical evaluate function as core function.

offical evaluate.py file can be find in the SQuAD dataset offcial web.
"""
import json

from mxnet import autograd, nd
from tqdm import tqdm

from data_loader import DataLoader
from offical_evaluate import evaluate as offical_eval

from config import CTX, opt

ctx = CTX[0]
ANSWER_MASK_MATRIX = nd.zeros(
    shape=(1, 1000, 1000), ctx=ctx)
for idx in range(opt.max_answer_len):
    ANSWER_MASK_MATRIX += nd.eye(
        N=1000, M=1000, k=idx, ctx=ctx)


def evaluate(model, dataset_type='train', ema=None):
    r"""Evaluate the model on train/dev/test dataset.

    This function is just an encapsulation of official evaluate function.

    The official evaluate code can be find in https://rajpurkar.github.io/SQuAD-explorer/

    Parameters
    ----------
    dataset_type : string, default 'train'
        which dataset to evaluate.
    ema : object or None, default None
        Whether use the shadow variable to evaluate.
    """
    model_cache_file_name = 'model_cache'
    model.save_parameters(model_cache_file_name)
    if ema is not None:
        for name, params in model.collect_params().items():
            params.set_data(ema.get(name))
    if dataset_type == 'train':
        data_loader = DataLoader(batch_size=opt.eval_batch_size, dev_set=False)
    else:
        data_loader = DataLoader(batch_size=opt.eval_batch_size, dev_set=True)
    autograd.set_training(False)
    total_answers = {}

    for batch_data in tqdm(data_loader.next_batch()):
        ids = [x[0] for x in batch_data]
        context = nd.array([x[1] for x in batch_data], ctx=ctx)
        query = nd.array([x[2] for x in batch_data], ctx=ctx)
        context_char = nd.array([x[3] for x in batch_data], ctx=ctx)
        query_char = nd.array([x[4] for x in batch_data], ctx=ctx)
        raw_context = [x[7] for x in batch_data]
        spans = [x[8] for x in batch_data]

        begin_hat, end_hat, _, _ = model(
            context, query,
            context_char,
            query_char,
            None,
            None
        )
        begin_hat = begin_hat.softmax(axis=1)
        end_hat = end_hat.softmax(axis=1)

        answer_span_pair = matrix_answer_select(begin_hat, end_hat)
        for i, a, r, s in zip(ids, answer_span_pair, raw_context, spans):
            total_answers[i] = format_answer(a, r, s)
    model.load_parameters(model_cache_file_name, ctx=CTX)
    autograd.set_training(True)
    if dataset_type == 'train':
        with open(opt.data_path + opt.train_file_name) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
    else:
        with open(opt.data_path + opt.dev_file_name) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
    result = offical_eval(dataset, total_answers)
    f1_score = result['f1']
    em_score = result['exact_match']
    return f1_score, em_score


def matrix_answer_select(begin_hat, end_hat):
    r"""Select the begin and end position of answer span.

        At inference time, the predicted span (s, e) is chosen such that
        begin_hat[s] * end_hat[e] is maximized and s ≤ e.

    Parameters
    ----------
    begin_hat : NDArray
        input tensor with shape `(batch_size, context_sequence_length)`
    end_hat : NDArray
        input tensor with shape `(batch_size, context_sequence_length)`
    """
    global ANSWER_MASK_MATRIX

    begin_hat = begin_hat.reshape(begin_hat.shape + (1,))
    end_hat = end_hat.reshape(end_hat.shape + (1,))
    end_hat = end_hat.transpose(axes=(0, 2, 1))

    result = nd.batch_dot(begin_hat, end_hat) * ANSWER_MASK_MATRIX.slice(
        begin=(0, 0, 0), end=(1, begin_hat.shape[1], begin_hat.shape[1]))
    yp1 = result.max(axis=2).argmax(axis=1, keepdims=True).astype('int32')
    yp2 = result.max(axis=1).argmax(axis=1, keepdims=True).astype('int32')
    return nd.concat(yp1, yp2, dim=-1)


def format_answer(answer_span_pair, context, sp):
    begin = int(answer_span_pair[0].asscalar())
    end = int(answer_span_pair[1].asscalar())
    return context[sp[begin][0]:sp[end][1]]
