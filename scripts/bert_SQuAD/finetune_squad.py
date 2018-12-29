"""
SQuAD with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
SQuAD, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import collections
import json
import logging
import os
import random
import time

import mxnet as mx
import numpy as np
from mxnet import gluon, nd

import gluonnlp as nlp
from bert import BERTloss, BERTSquad
from dataset import (SQuAD, SQuADTransform, bert_qa_batchify_fn,
                     preprocess_dataset)
from evaluate import evaluate, predictions

np.random.seed(0)
random.seed(0)
mx.random.seed(2)
logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='BERT QA example.'
                                 'We fine-tune the BERT model on SQuAD 1.1')

parser.add_argument(
    '--train_file',
    type=str,
    default='train-v1.1.json',
    help='SQuAD json for training. E.g., train-v1.1.json')
parser.add_argument(
    '--predict_file',
    type=str,
    default=None,
    help='SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json')
parser.add_argument(
    '--output_dir',
    type=str,
    default='./output_dir',
    help='The output directory where the model params will be written. default is ./output_dir')

parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs, default is 2')

parser.add_argument(
    '--batch_size',
    type=int,
    default=12,
    help='Batch size. Number of examples per gpu in a minibatch. default is 12')

parser.add_argument(
    '--test_batch_size', type=int, default=24, help='Test batch size. default is 24')

parser.add_argument(
    '--optimizer', type=str, default='adam', help='optimization algorithm. default is adam)
parser.add_argument(
    '--lr', type=float, default=3e-5, help='Initial learning rate. default is 3e-5')

parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0.1,
    help='ratio of warmup steps used in NOAM\'s stepsize schedule. default is 0.1')

parser.add_argument(
    '--log_interval', type=int, default=50, help='report interval. default is 50')

parser.add_argument(
    '--max_seq_length',
    type=int,
    default=384,
    help='The maximum total input sequence length after WordPiece tokenization.'
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. default is 384')

parser.add_argument(
    '--doc_stride',
    type=int,
    default=128,
    help='When splitting up a long document into chunks, how much stride to '
    'take between chunks. default is 128')

parser.add_argument(
    '--max_query_length',
    type=int,
    default=64,
    help='The maximum number of tokens for the question. Questions longer than '
    'this will be truncated to this length. default is 64')

parser.add_argument(
    '--n_best_size',
    type=int,
    default=20,
    help='The total number of n-best predictions to generate in the '
    'nbest_predictions.json output file. default is 20')

parser.add_argument(
    '--max_answer_length',
    type=int,
    default=30,
    help='The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another. default is 30'
)

parser.add_argument(
    '--version_2',
    type=bool,
    default=False,
    help='If true, the SQuAD examples contain some that do not have an answer. default is False'
)

parser.add_argument(
    '--null_score_diff_threshold',
    type=float,
    default=0.0,
    help='If null_score - best_non_null is greater than the threshold predict null. default is 0.0'
)

parser.add_argument(
    '--gpu', action='store_true', help='whether to use gpu for finetuning')

args = parser.parse_args()
logging.info(args)

train_file = args.train_file
predict_file = args.predict_file
output_dir = args.output_dir
if os.path.exists(output_dir):
    os.mkdir(output_dir)

epochs = args.epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size
lr = args.lr
ctx = mx.cpu() if not args.gpu else mx.gpu()
optimizer = args.optimizer
log_interval = args.log_interval
warmup_ratio = args.warmup_ratio

dataset_name = 'book_corpus_wiki_en_uncased'
version_2 = args.version_2
max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length
n_best_size = args.n_best_size
max_answer_length = args.max_answer_length
null_score_diff_threshold = args.null_score_diff_threshold

if max_seq_length <= max_query_length + 3:
    raise ValueError(
        'The max_seq_length (%d) must be greater than max_query_length '
        '(%d) + 3' % (max_seq_length, max_query_length))

bert, vocab = nlp.model.bert_12_768_12(
    dataset_name=dataset_name,
    pretrained=True,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False,
)

berttoken = nlp.data.BERTTokenizer(vocab=vocab)


logging.info('Loader Train data...')
train_data = SQuAD(train_file, version_2=version_2)

train_data_transform = preprocess_dataset(train_data, SQuADTransform(
    berttoken,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=True))

train_dataloader = mx.gluon.data.DataLoader(
    train_data_transform, batch_size=batch_size, batchify_fn=bert_qa_batchify_fn, num_workers=4, shuffle=True)

net = BERTSquad(bert=bert)
net.Dense.initialize(init=mx.init.Normal(0.02), ctx=ctx)
net.hybridize(static_alloc=True)

loss_function = BERTloss()
loss_function.hybridize(static_alloc=True)


def Train():

    logging.info('Start Training')

    trainer = gluon.Trainer(net.collect_params(), optimizer, {
        'learning_rate': lr,
        'epsilon': 1e-9
    })

    num_train_examples = len(train_data)
    num_train_steps = int(num_train_examples / batch_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    differentiable_params = []

    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in net.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    for epoch_id in range(epochs):
        step_loss = 0.0
        tic = time.time()
        for batch_id, data in enumerate(train_dataloader):

            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / \
                    (num_train_steps - num_warmup_steps)
                new_lr = lr - offset
            trainer.set_learning_rate(new_lr)

            with mx.autograd.record():
                _, inputs, token_types, valid_length, start_label, end_label = data

                out = net(
                    inputs.astype('float32').as_in_context(ctx),
                    token_types.astype('float32').as_in_context(ctx),
                    valid_length.astype('float32').as_in_context(ctx))

                ls = loss_function(out, [
                    start_label.astype('float32').as_in_context(ctx),
                    end_label.astype('float32').as_in_context(ctx)
                ]).mean()
            ls.backward()

            grads = [p.grad(ctx) for p in differentiable_params]
            gluon.utils.clip_global_norm(grads, 1)
            trainer.step(1)

            step_loss += ls.asscalar()

            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                logging.info(
                    'Epoch: %d, Batch: %d/%d, Loss=%.4f, lr=%.7f Time cost=%.1f'
                    % (epoch_id, batch_id, len(train_dataloader), step_loss /
                       (log_interval), trainer.learning_rate, toc - tic))
                tic = time.time()
                step_loss = 0.0

    net.save_parameters(output_dir + 'net_parameters')


def Evaluate():
    logging.info('Loader dev data...')
    dev_data = SQuAD(predict_file, version_2=version_2, is_training=False)

    dev_dataset = dev_data.transform(
        SQuADTransform(
            berttoken,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False)._transform)

    dev_data_transform = preprocess_dataset(dev_data, SQuADTransform(
        berttoken,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False))

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform, batch_size=test_batch_size, batchify_fn=bert_qa_batchify_fn, num_workers=4, shuffle=False, last_batch='keep')

    start_logits = []
    end_logits = []
    logging.info('Start predict')

    _Result = collections.namedtuple(
        "_Result", ["example_id", "start_logits", "end_logits"])
    all_results = {}

    for data in dev_dataloader:
        example_ids, inputs, token_types, valid_length, _, _ = data

        out = net(
            inputs.astype('float32').as_in_context(ctx),
            token_types.astype('float32').as_in_context(ctx),
            valid_length.astype('float32').as_in_context(ctx))

        output = nd.split(out, axis=0, num_outputs=2)
        start_logits = output[0].reshape((-3, 0)).asnumpy()
        end_logits = output[1].reshape((-3, 0)).asnumpy()

        for example_id, start, end in zip(example_ids, start_logits, end_logits):
            example_id = example_id.asscalar()
            if example_id not in all_results:
                all_results[example_id] = []
            all_results[example_id].append(
                _Result(example_id, start.tolist(), end.tolist()))

    all_predictions, all_nbest_json, scores_diff_json = predictions(
        dev_dataset=dev_dataset,
        all_results=all_results,
        max_answer_length=max_answer_length,
        tokenizer=nlp.data.BasicTokenizer(lower_case=True))

    with open(
            os.path.join(output_dir, 'predictions.json'), 'w',
            encoding='utf-8') as all_predictions_write:
        all_predictions_write.write(json.dumps(all_predictions))

    with open(
            os.path.join(output_dir, 'nbest_predictions.json'),
            'w',
            encoding='utf-8') as all_predictions_write:
        all_predictions_write.write(json.dumps(all_nbest_json))

    if version_2:
        with open(
                os.path.join(output_dir, 'null_odds.json'), 'w',
                encoding='utf-8') as all_predictions_write:
            all_predictions_write.write(json.dumps(scores_diff_json))
    else:
        logging.info(evaluate(predict_file, all_predictions))


if __name__ == '__main__':
    Train()
    if predict_file is not None:
        Evaluate()
