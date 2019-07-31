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

# coding=utf-8

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
import io
import copy
import random
import time
import warnings

import numpy as np
import mxnet as mx

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from model.qa import BertForQALoss, BertForQA
from data.qa import SQuADTransform, preprocess_dataset
from bert_qa_evaluate import get_F1_EM, predict, PredResult

np.random.seed(6)
random.seed(6)
mx.random.seed(6)

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')


parser = argparse.ArgumentParser(description='BERT QA example.'
                                 'We fine-tune the BERT model on SQuAD dataset.')

parser.add_argument('--only_predict',
                    action='store_true',
                    help='Whether to predict only.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.')

parser.add_argument('--bert_dataset',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    help='BERT dataset name.'
                    'options are book_corpus_wiki_en_uncased and book_corpus_wiki_en_cased.')

parser.add_argument('--pretrained_bert_parameters',
                    type=str,
                    default=None,
                    help='Pre-trained bert model parameter file. default is None')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The output directory where the model params will be written.'
                    ' default is ./output_dir')

parser.add_argument('--epochs',
                    type=int,
                    default=3,
                    help='number of epochs, default is 3')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer',
                    type=str,
                    default='bertadam',
                    help='optimization algorithm. default is bertadam(mxnet >= 1.5.0.)')

parser.add_argument('--accumulate',
                    type=int,
                    default=None,
                    help='The number of batches for '
                    'gradients accumulation to simulate large batch size. Default is None')

parser.add_argument('--lr',
                    type=float,
                    default=5e-5,
                    help='Initial learning rate. default is 5e-5')

parser.add_argument('--warmup_ratio',
                    type=float,
                    default=0.1,
                    help='ratio of warmup steps that linearly increase learning rate from '
                    '0 to target learning rate. default is 0.1')

parser.add_argument('--log_interval',
                    type=int,
                    default=50,
                    help='report interval. default is 50')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                    'Sequences longer than this will be truncated, and sequences shorter '
                    'than this will be padded. default is 384')

parser.add_argument('--doc_stride',
                    type=int,
                    default=128,
                    help='When splitting up a long document into chunks, how much stride to '
                    'take between chunks. default is 128')

parser.add_argument('--max_query_length',
                    type=int,
                    default=64,
                    help='The maximum number of tokens for the question. Questions longer than '
                    'this will be truncated to this length. default is 64')

parser.add_argument('--n_best_size',
                    type=int,
                    default=20,
                    help='The total number of n-best predictions to generate in the '
                    'nbest_predictions.json output file. default is 20')

parser.add_argument('--max_answer_length',
                    type=int,
                    default=30,
                    help='The maximum length of an answer that can be generated. This is needed '
                    'because the start and end predictions are not conditioned on one another.'
                    ' default is 30')

parser.add_argument('--version_2',
                    action='store_true',
                    help='SQuAD examples whether contain some that do not have an answer.')

parser.add_argument('--null_score_diff_threshold',
                    type=float,
                    default=0.0,
                    help='If null_score - best_non_null is greater than the threshold predict null.'
                    'Typical values are between -1.0 and -5.0. default is 0.0')

parser.add_argument('--gpu',
                    type=int,
                    default=None,
                    help='which gpu to use for finetuning. CPU is used if not set.')

parser.add_argument('--sentencepiece',
                    type=str,
                    default=None,
                    help='Path to the sentencepiece .model file for both tokenization and vocab.')

parser.add_argument('--debug',
                    action='store_true',
                    help='Run the example in test mode for sanity checks')

args = parser.parse_args()

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fh = logging.FileHandler(os.path.join(
    args.output_dir, 'finetune_squad.log'), mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

log.info(args)

model_name = args.bert_model
dataset_name = args.bert_dataset
only_predict = args.only_predict
model_parameters = args.model_parameters
pretrained_bert_parameters = args.pretrained_bert_parameters
if pretrained_bert_parameters and model_parameters:
    raise ValueError('Cannot provide both pre-trained BERT parameters and '
                     'BertForQA model parameters.')
lower = args.uncased

epochs = args.epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size
lr = args.lr
ctx = mx.cpu() if args.gpu is None else mx.gpu(args.gpu)

accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    log.info('Using gradient accumulation. Effective batch size = {}'.
             format(accumulate*batch_size))

optimizer = args.optimizer
warmup_ratio = args.warmup_ratio


version_2 = args.version_2
null_score_diff_threshold = args.null_score_diff_threshold

max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length
n_best_size = args.n_best_size
max_answer_length = args.max_answer_length

if max_seq_length <= max_query_length + 3:
    raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                     '(%d) + 3' % (max_seq_length, max_query_length))

# vocabulary and tokenizer
if args.sentencepiece:
    logging.info('loading vocab file from sentence piece model: %s', args.sentencepiece)
    if dataset_name:
        warnings.warn('Both --dataset_name and --sentencepiece are provided. '
                      'The vocabulary will be loaded based on --sentencepiece.')
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(args.sentencepiece)
    dataset_name = None
else:
    vocab = None

pretrained = not model_parameters and not pretrained_bert_parameters and not args.sentencepiece
bert, vocab = nlp.model.get_model(
    name=model_name,
    dataset_name=dataset_name,
    vocab=vocab,
    pretrained=pretrained,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False)

if args.sentencepiece:
    tokenizer = nlp.data.BERTSPTokenizer(args.sentencepiece, vocab, lower=lower)
else:
    tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'))

net = BertForQA(bert=bert)
if model_parameters:
    # load complete BertForQA parameters
    nlp.utils.load_parameters(net, model_parameters, ctx=ctx, cast_dtype=True)
elif pretrained_bert_parameters:
    # only load BertModel parameters
    nlp.utils.load_parameters(bert, pretrained_bert_parameters, ctx=ctx,
                              ignore_extra=True, cast_dtype=True)
    net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
elif pretrained:
    # only load BertModel parameters
    net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
else:
    # no checkpoint is loaded
    net.initialize(init=mx.init.Normal(0.02), ctx=ctx)

net.hybridize(static_alloc=True)

loss_function = BertForQALoss()
loss_function.hybridize(static_alloc=True)


def train():
    """Training function."""
    segment = 'train' if not args.debug else 'dev'
    log.info('Loading %s data...', segment)
    if version_2:
        train_data = SQuAD(segment, version='2.0')
    else:
        train_data = SQuAD(segment, version='1.1')
    if args.debug:
        sampled_data = [train_data[i] for i in range(1000)]
        train_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in Train data:{}'.format(len(train_data)))

    train_data_transform, _ = preprocess_dataset(
        train_data, SQuADTransform(
            copy.copy(tokenizer),
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=True,
            is_training=True))
    log.info('The number of examples after preprocessing:{}'.format(
        len(train_data_transform)))

    train_dataloader = mx.gluon.data.DataLoader(
        train_data_transform, batchify_fn=batchify_fn,
        batch_size=batch_size, num_workers=4, shuffle=True)

    log.info('Start Training')

    optimizer_params = {'learning_rate': lr}
    try:
        trainer = mx.gluon.Trainer(net.collect_params(), optimizer,
                                   optimizer_params, update_on_kvstore=False)
    except ValueError as e:
        print(e)
        warnings.warn('AdamW optimizer is not found. Please consider upgrading to '
                      'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')
        trainer = mx.gluon.Trainer(net.collect_params(), 'adam',
                                   optimizer_params, update_on_kvstore=False)

    num_train_examples = len(train_data_transform)
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    def set_new_lr(step_num, batch_id):
        """set new learning rate"""
        # set grad to zero for gradient accumulation
        if accumulate:
            if batch_id % accumulate == 0:
                net.collect_params().zero_grad()
                step_num += 1
        else:
            step_num += 1
        # learning rate schedule
        # Notice that this learning rate scheduler is adapted from traditional linear learning
        # rate scheduler where step_num >= num_warmup_steps, new_lr = 1 - step_num/num_train_steps
        if step_num < num_warmup_steps:
            new_lr = lr * step_num / num_warmup_steps
        else:
            offset = (step_num - num_warmup_steps) * lr / \
                (num_train_steps - num_warmup_steps)
            new_lr = lr - offset
        trainer.set_learning_rate(new_lr)
        return step_num

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in net.collect_params().values()
              if p.grad_req != 'null']
    # Set grad_req if gradient accumulation is required
    if accumulate:
        for p in params:
            p.grad_req = 'add'

    epoch_tic = time.time()
    total_num = 0
    log_num = 0
    for epoch_id in range(epochs):
        step_loss = 0.0
        tic = time.time()
        for batch_id, data in enumerate(train_dataloader):
            # set new lr
            step_num = set_new_lr(step_num, batch_id)
            # forward and backward
            with mx.autograd.record():
                _, inputs, token_types, valid_length, start_label, end_label = data

                log_num += len(inputs)
                total_num += len(inputs)

                out = net(inputs.astype('float32').as_in_context(ctx),
                          token_types.astype('float32').as_in_context(ctx),
                          valid_length.astype('float32').as_in_context(ctx))

                ls = loss_function(out, [
                    start_label.astype('float32').as_in_context(ctx),
                    end_label.astype('float32').as_in_context(ctx)]).mean()

                if accumulate:
                    ls = ls / accumulate
            ls.backward()
            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1)

            step_loss += ls.asscalar()

            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                log.info('Epoch: {}, Batch: {}/{}, Loss={:.4f}, lr={:.7f} Time cost={:.1f} Thoughput={:.2f} samples/s'  # pylint: disable=line-too-long
                         .format(epoch_id, batch_id, len(train_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, toc - tic, log_num/(toc - tic)))
                tic = time.time()
                step_loss = 0.0
                log_num = 0
        epoch_toc = time.time()
        log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
            epoch_toc - epoch_tic, total_num/(epoch_toc - epoch_tic)))

    net.save_parameters(os.path.join(output_dir, 'net.params'))


def evaluate():
    """Evaluate the model on validation dataset.
    """
    log.info('Loading dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    if args.debug:
        sampled_data = [dev_data[0], dev_data[1], dev_data[2]]
        dev_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in dev data:{}'.format(len(dev_data)))

    dev_dataset = dev_data.transform(
        SQuADTransform(
            copy.copy(tokenizer),
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=False,
            is_training=False)._transform, lazy=False)

    dev_data_transform, _ = preprocess_dataset(
        dev_data, SQuADTransform(
            copy.copy(tokenizer),
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=False,
            is_training=False))
    log.info('The number of examples after preprocessing:{}'.format(
        len(dev_data_transform)))

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn,
        num_workers=4, batch_size=test_batch_size,
        shuffle=False, last_batch='keep')

    log.info('start prediction')

    all_results = collections.defaultdict(list)

    epoch_tic = time.time()
    total_num = 0
    for data in dev_dataloader:
        example_ids, inputs, token_types, valid_length, _, _ = data
        total_num += len(inputs)
        out = net(inputs.astype('float32').as_in_context(ctx),
                  token_types.astype('float32').as_in_context(ctx),
                  valid_length.astype('float32').as_in_context(ctx))

        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()

        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            all_results[example_id].append(PredResult(start=start, end=end))

    epoch_toc = time.time()
    log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
        epoch_toc - epoch_tic, total_num/(epoch_toc - epoch_tic)))

    log.info('Get prediction results...')

    all_predictions = collections.OrderedDict()

    for features in dev_dataset:
        results = all_results[features[0].example_id]
        example_qas_id = features[0].qas_id

        prediction, _ = predict(
            features=features,
            results=results,
            tokenizer=nlp.data.BERTBasicTokenizer(lower=lower),
            max_answer_length=max_answer_length,
            null_score_diff_threshold=null_score_diff_threshold,
            n_best_size=n_best_size,
            version_2=version_2)

        all_predictions[example_qas_id] = prediction

    with io.open(os.path.join(output_dir, 'predictions.json'),
                 'w', encoding='utf-8') as fout:
        data = json.dumps(all_predictions, ensure_ascii=False)
        fout.write(data)

    if version_2:
        log.info('Please run evaluate-v2.0.py to get evaluation results for SQuAD 2.0')
    else:
        F1_EM = get_F1_EM(dev_data, all_predictions)
        log.info(F1_EM)


if __name__ == '__main__':
    if not only_predict:
        train()
        evaluate()
    elif model_parameters:
        evaluate()
