"""
Export SQuAD with Static Bidirectional Encoder Representations from Transformers (BERT)

=========================================================================================

This example shows how to export a Block based BERT model with pre-trained BERT parameters
with static shape, we are using SQuAD as an example.

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
import logging
import os
import time

import mxnet as mx

from static_bert_qa_model import StaticBertForQA
from static_bert import get_model

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(description='export static BERT QA example.')

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

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

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

parser.add_argument('--gpu', type=str, help='single gpu id')

parser.add_argument('--seq_length',
                    type=int,
                    default=384,
                    help='The sequence length of the input')

parser.add_argument('--input_size',
                    type=int,
                    default=768,
                    help='The embedding size of the input')

parser.add_argument('--export',
                    action='store_true',
                    help='Whether to export the model.')

parser.add_argument('--evaluate',
                    action='store_true',
                    help='Whether to evaluate the model.')

args = parser.parse_args()


output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fh = logging.FileHandler(os.path.join(
    args.output_dir, 'static_export_squad.log'), mode='w')
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
model_parameters = args.model_parameters
pretrained_bert_parameters = args.pretrained_bert_parameters
lower = args.uncased

seq_length = args.seq_length
input_size = args.input_size
test_batch_size = args.test_batch_size
ctx = mx.cpu() if not args.gpu else mx.gpu(int(args.gpu))

max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length

if max_seq_length <= max_query_length + 3:
    raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                     '(%d) + 3' % (max_seq_length, max_query_length))


###############################################################################
#                              Prepare dummy input data                       #
###############################################################################
if args.evaluate:
    inputs = mx.nd.arange(test_batch_size * seq_length).reshape(shape=(test_batch_size, seq_length))
    token_types = mx.nd.zeros_like(inputs)
    valid_length = mx.nd.arange(seq_length)[:test_batch_size]
    batch = inputs, token_types, valid_length
    num_batch = 10
    sample_dataset = []
    for _ in range(num_batch):
        sample_dataset.append(batch)


bert, vocab = get_model(
    name=model_name,
    dataset_name=dataset_name,
    pretrained=not model_parameters and not pretrained_bert_parameters,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False,
    input_size=args.input_size,
    seq_length=args.seq_length)


###############################################################################
#                              Hybridize the model                            #
###############################################################################
net = StaticBertForQA(bert=bert)
if pretrained_bert_parameters and not model_parameters:
    bert.load_parameters(pretrained_bert_parameters, ctx=ctx,
                         ignore_extra=True)
if not model_parameters:
    net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
else:
    net.load_parameters(model_parameters, ctx=ctx)

net.hybridize(static_alloc=True, static_shape=True)


def evaluate(data_source):
    """Evaluate the model on a mini-batch.
    """
    log.info('Start predict')
    tic = time.time()
    for batch in data_source:
        inputs, token_types, valid_length = batch
        out = net(inputs.astype('float32').as_in_context(ctx),
                  token_types.astype('float32').as_in_context(ctx),
                  valid_length.astype('float32').as_in_context(ctx))
    toc = time.time()
    log.info('Inference time cost={:.2f} s, Thoughput={:.2f} samples/s'
             .format(toc - tic,
                     len(data_source) / (toc - tic)))



###############################################################################
#                              Export the model                               #
###############################################################################
if __name__ == '__main__':
    if args.export:
        net.export(os.path.join(args.output_dir, 'static_net'), epoch=0)
        if args.evaluate:
            net.load_parameters(os.path.join(args.output_dir, 'static_net-0000.params'))
            evaluate(sample_dataset)
    else:
        if args.evaluate:
            evaluate(sample_dataset)
