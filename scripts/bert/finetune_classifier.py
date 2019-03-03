"""
Sentence Pair Classification with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
sentence pair classification, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
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

import os
import time
import argparse
import random
import logging
import warnings
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from gluonnlp.model import get_bert_model
from gluonnlp.data import BERTTokenizer

from bert import BERTClassifier, BERTRegression
from dataset import MRPCDataset, QQPDataset, RTEDataset, \
    STSBDataset, BERTDatasetTransform, \
    QNLIDataset, COLADataset, MNLIDataset, WNLIDataset, SSTDataset

tasks = {
    'MRPC': MRPCDataset,
    'QQP': QQPDataset,
    'QNLI': QNLIDataset,
    'RTE': RTEDataset,
    'STS-B': STSBDataset,
    'CoLA': COLADataset,
    'MNLI': MNLIDataset,
    'WNLI': WNLIDataset,
    'SST': SSTDataset
}

parser = argparse.ArgumentParser(
    description='BERT fine-tune examples for GLUE tasks.')
parser.add_argument(
    '--epochs', type=int, default=3, help='number of epochs, default is 3')
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size. Number of examples per gpu in a minibatch, default is 32'
)
parser.add_argument(
    '--dev_batch_size',
    type=int,
    default=8,
    help='Batch size for dev set, default is 8')
parser.add_argument(
    '--optimizer',
    type=str,
    default='bertadam',
    help='Optimization algorithm, default is bertadam')
parser.add_argument(
    '--lr',
    type=float,
    default=5e-5,
    help='Initial learning rate, default is 5e-5')
parser.add_argument(
    '--epsilon',
    type=float,
    default=1e-06,
    help='Small value to avoid division by 0, default is 1e-06'
)
parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0.1,
    help='ratio of warmup steps used in NOAM\'s stepsize schedule, default is 0.1')
parser.add_argument(
    '--log_interval',
    type=int,
    default=10,
    help='report interval, default is 10')
parser.add_argument(
    '--max_len',
    type=int,
    default=128,
    help='Maximum length of the sentence pairs, default is 128')
parser.add_argument(
    '--seed', type=int, default=2, help='Random seed, default is 2')
parser.add_argument(
    '--accumulate',
    type=int,
    default=None,
    help='The number of batches for '
    'gradients accumulation to simulate large batch size. Default is None')
parser.add_argument(
    '--gpu', action='store_true', help='whether to use gpu for finetuning')
parser.add_argument(
    '--task_name',
    type=str,
    choices=tasks.keys(),
    help='The name of the task to fine-tune.(MRPC,...)')
parser.add_argument(
    '--bert_model',
    type=str,
    default='bert_12_768_12',
    help='The name of pre-trained BERT model to fine-tune'
    '(bert_24_1024_16 and bert_12_768_12).')
parser.add_argument(
    '--bert_dataset',
    type=str,
    default='book_corpus_wiki_en_uncased',
    help='Dataset of BERT pre-trained with.'
    'Options include \'book_corpus_wiki_en_cased\', \'book_corpus_wiki_en_uncased\''
    'for both bert_24_1024_16 and bert_12_768_12.'
    '\'wiki_cn_cased\', \'wiki_multilingual_uncased\' and \'wiki_multilingual_cased\''
    'for bert_12_768_12 only.'
)
parser.add_argument(
    '--pretrained_bert_parameters',
    type=str,
    default=None,
    help='Pre-trained bert model parameter file. default is None'
)
parser.add_argument(
    '--model_parameters',
    type=str,
    default=None,
    help='A parameter file for the model that is loaded into the model'
    ' before training/inference. It is different from the parameter'
    ' file written after the model is trained. default is None'
)
parser.add_argument(
    '--output_dir',
    type=str,
    default='./output_dir',
    help='The output directory where the model params will be written.'
    ' default is ./output_dir'
)

args = parser.parse_args()

logging.getLogger().setLevel(logging.DEBUG)
logging.captureWarnings(True)
logging.info(args)

batch_size = args.batch_size
dev_batch_size = args.dev_batch_size
task_name = args.task_name
lr = args.lr
epsilon = args.epsilon
accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    logging.info('Using gradient accumulation. Effective batch size = %d',
                 accumulate * batch_size)

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)

ctx = mx.cpu() if not args.gpu else mx.gpu()

task = tasks[task_name]

# model and loss
model_name = args.bert_model
dataset = args.bert_dataset
pretrained_bert_parameters = args.pretrained_bert_parameters
model_parameters = args.model_parameters

get_pretrained = not (pretrained_bert_parameters is not None
                      or model_parameters is not None)
bert, vocabulary = get_bert_model(
    model_name=model_name,
    dataset_name=dataset,
    pretrained=get_pretrained,
    ctx=ctx,
    use_pooler=True,
    use_decoder=False,
    use_classifier=False)

if task.task_name in ['STS-B']:
    model = BERTRegression(bert, dropout=0.1)
    if not model_parameters:
        model.regression.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    loss_function = gluon.loss.L2Loss()
else:
    model = BERTClassifier(
        bert, dropout=0.1, num_classes=len(task.get_labels()))
    if not model_parameters:
        model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    loss_function = gluon.loss.SoftmaxCELoss()

# load checkpointing
output_dir = args.output_dir
if pretrained_bert_parameters:
    logging.info('loading bert params from {0}'.format(pretrained_bert_parameters))
    model.bert.load_parameters(pretrained_bert_parameters, ctx=ctx,
                               ignore_extra=True)
if model_parameters:
    logging.info('loading model params from {0}'.format(model_parameters))
    model.load_parameters(model_parameters, ctx=ctx)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.info(model)
model.hybridize(static_alloc=True)
loss_function.hybridize(static_alloc=True)

# data processing
do_lower_case = 'uncased' in dataset
bert_tokenizer = BERTTokenizer(vocabulary, lower=do_lower_case)


def preprocess_data(tokenizer, task, batch_size, dev_batch_size, max_len):
    """Data preparation function."""
    # transformation
    trans = BERTDatasetTransform(
        tokenizer,
        max_len,
        labels=task.get_labels(),
        pad=False,
        pair=task.is_pair,
        label_dtype='float32' if not task.get_labels() else 'int32')

    data_train = task('train').transform(trans, lazy=False)
    data_train_len = data_train.transform(
        lambda input_id, length, segment_id, label_id: length)

    num_samples_train = len(data_train)
    # bucket sampler
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0),
        nlp.data.batchify.Stack(
            'float32' if not task.get_labels() else 'int32'))
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        data_train_len,
        batch_size=batch_size,
        num_buckets=10,
        ratio=0,
        shuffle=True)
    # data loaders
    dataloader_train = gluon.data.DataLoader(
        dataset=data_train,
        num_workers=1,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)
    if task.task_name == 'MNLI':
        data_dev_matched = task('dev_matched').transform(trans, lazy=False)
        data_dev_mismatched = task('dev_mismatched').transform(trans, lazy=False)

        dataloader_dev_matched = mx.gluon.data.DataLoader(
            data_dev_matched, batch_size=dev_batch_size,
            num_workers=1, shuffle=False, batchify_fn=batchify_fn)
        dataloader_dev_mismatched = mx.gluon.data.DataLoader(
            data_dev_mismatched, batch_size=dev_batch_size,
            num_workers=1, shuffle=False, batchify_fn=batchify_fn)
        return dataloader_train, dataloader_dev_matched, \
            dataloader_dev_mismatched, num_samples_train
    else:
        data_dev = task('dev').transform(trans, lazy=False)
        dataloader_dev = mx.gluon.data.DataLoader(
            data_dev,
            batch_size=dev_batch_size,
            num_workers=1,
            shuffle=False,
            batchify_fn=batchify_fn)
        return dataloader_train, dataloader_dev, num_samples_train


# Get the dataloader. Data set for special handling of MNLI tasks
logging.info('processing dataset...')
if task.task_name == 'MNLI':
    train_data, dev_data_matched, dev_data_mismatched, num_train_examples = preprocess_data(
        bert_tokenizer, task, batch_size, dev_batch_size, args.max_len)
else:
    train_data, dev_data, num_train_examples = preprocess_data(
        bert_tokenizer, task, batch_size, dev_batch_size, args.max_len)


def evaluate(dataloader_eval, metric):
    """Evaluate the model on validation dataset.
    """
    metric.reset()
    for _, seqs in enumerate(dataloader_eval):
        input_ids, valid_len, type_ids, label = seqs
        out = model(
            input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
            valid_len.astype('float32').as_in_context(ctx))
        metric.update([label], [out])
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm = [metric_nm]
        metric_val = [metric_val]
    metric_str = 'validation metrics:' + ','.join(
        [i + ':%.4f' for i in metric_nm])
    logging.info(metric_str, *metric_val)


def train(metric):
    """Training function."""
    optimizer_params = {'learning_rate': lr, 'epsilon': epsilon, 'wd': 0.01}
    try:
        trainer = gluon.Trainer(
            model.collect_params(),
            args.optimizer,
            optimizer_params,
            update_on_kvstore=False)
    except ValueError as e:
        print(e)
        warnings.warn(
            'AdamW optimizer is not found. Please consider upgrading to '
            'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')
        trainer = gluon.Trainer(
            model.collect_params(),
            'adam',
            optimizer_params,
            update_on_kvstore=False)

    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [
        p for p in model.collect_params().values() if p.grad_req != 'null'
    ]
    # Set grad_req if gradient accumulation is required
    if accumulate:
        for p in params:
            p.grad_req = 'add'

    for epoch_id in range(args.epochs):
        metric.reset()
        step_loss = 0
        tic = time.time()
        for batch_id, seqs in enumerate(train_data):
            # set grad to zero for gradient accumulation
            if accumulate:
                if batch_id % accumulate == 0:
                    model.collect_params().zero_grad()
                    step_num += 1
            else:
                step_num += 1
            # learning rate schedule
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / (
                    num_train_steps - num_warmup_steps)
                new_lr = lr - offset
            trainer.set_learning_rate(new_lr)
            # forward and backward
            with mx.autograd.record():
                input_ids, valid_length, type_ids, label = seqs
                out = model(
                    input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                    valid_length.astype('float32').as_in_context(ctx))
                ls = loss_function(out, label.as_in_context(ctx)).mean()
            ls.backward()
            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(accumulate if accumulate else 1)
            step_loss += ls.asscalar()
            metric.update([label], [out])
            if (batch_id + 1) % (args.log_interval) == 0:
                metric_nm, metric_val = metric.get()
                if not isinstance(metric_nm, list):
                    metric_nm = [metric_nm]
                    metric_val = [metric_val]
                eval_str = '[Epoch %d Batch %d/%d] loss=%.4f, lr=%.7f, metrics=' + \
                    ','.join([i + ':%.4f' for i in metric_nm])
                logging.info(eval_str, epoch_id + 1, batch_id + 1, len(train_data),
                             step_loss / args.log_interval,
                             trainer.learning_rate, *metric_val)
                step_loss = 0
        mx.nd.waitall()
        if task.task_name == 'MNLI':
            logging.info('On MNLI Matched: ')
            evaluate(dev_data_matched, metric)
            logging.info('On MNLI Mismatched: ')
            evaluate(dev_data_mismatched, metric)
        else:
            evaluate(dev_data, metric)

        # save params
        params_saved = os.path.join(output_dir,
                                    'model_bert_{0}_{1}.params'.format(task.task_name, epoch_id))
        model.save_parameters(params_saved)
        logging.info('params saved in : {0}'.format(params_saved))
        toc = time.time()
        logging.info('Time cost=%.1fs', toc - tic)
        tic = toc


if __name__ == '__main__':
    pool_type = os.environ.get('MXNET_GPU_MEM_POOL_TYPE', '')
    if pool_type.lower() == 'round':
        logging.info(
            'Setting MXNET_GPU_MEM_POOL_TYPE="Round" may lead to higher memory '
            'usage and faster speed. If you encounter OOM errors, please unset '
            'this environment variable.')
    train(task.get_metric())
