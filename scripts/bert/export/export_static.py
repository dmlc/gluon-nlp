"""
Export the BERT Model for Deployment

====================================

This script exports the BERT model to a static model suitable for use with MXNet Module API.

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
import warnings
import os
import time

import mxnet as mx
import gluonnlp as nlp
from static_bert import get_model

parser = argparse.ArgumentParser(description='Export static BERT base model.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='The model parameter file saved from training.')

parser.add_argument('--model_name',
                    type=str,
                    default='bert_12_768_12',
                    choices=['bert_12_768_12', 'bert_24_1024_16'],
                    help='BERT model name. Options are "bert_12_768_12" and "bert_24_1024_16"')

parser.add_argument('--task',
                    type=str,
                    default=None,
                    choices=['classification', 'regression', 'qa'],
                    help='Task to export. Options are "classification", "regression", "qa". '
                         'If not set, the model for masked language model and next sentence '
                         'prediction will be exported.')

parser.add_argument('--dataset_name',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                             'wiki_multilingual_uncased', 'wiki_multilingual_cased',
                             'wiki_cn_cased'],
                    help='BERT dataset name. Options include '
                         '"book_corpus_wiki_en_uncased", "book_corpus_wiki_en_cased", '
                         '"wiki_multilingual_uncased", "wiki_multilingual_cased", '
                         '"wiki_cn_cased"')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The directory where the exported model symbol will be created. '
                         'The default is ./output_dir')

parser.add_argument('--seq_length',
                    type=int,
                    default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                         'Sequences longer than this needs to be truncated, and sequences shorter '
                         'than this needs to be padded. Default is 384')

args = parser.parse_args()

# create output dir
output_dir = args.output_dir
nlp.utils.mkdir(output_dir)

# logging
log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s',
                              datefmt='%H:%M:%S')
fh = logging.FileHandler(os.path.join(args.output_dir, 'static_export_bert.log'), mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

log.info(args)

model_parameters = args.model_parameters
seq_length = args.seq_length
test_batch_size = 1
ctx = mx.cpu()

###############################################################################
#                              Prepare dummy input data                       #
###############################################################################

inputs = mx.nd.arange(test_batch_size * seq_length)
inputs = inputs.reshape(shape=(test_batch_size, seq_length))
token_types = mx.nd.zeros_like(inputs)
valid_length = mx.nd.arange(test_batch_size)
batch = inputs, token_types, valid_length
num_batch = 10
sample_dataset = [batch for _ in range(10)]

bert, vocab = get_model(
    name=args.model_name,
    dataset_name=args.dataset_name,
    pretrained=True,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False,
    seq_length=args.seq_length)


###############################################################################
#                              Hybridize the model                            #
###############################################################################
net = bert
if args.task == 'classification':
    net = StaticBERTClassifier(net, num_classes=2)

if model_parameters:
    bert.load_parameters(model_parameters, ctx=ctx)
else:
    warnings.warn('using random initialization')

net.hybridize(static_alloc=True, static_shape=True)

def evaluate(data_source):
    """Evaluate the model on a mini-batch."""
    log.info('start predicting ... ')
    tic = time.time()
    for inputs, token_types, valid_length in data_source:
        net(inputs.as_in_context(ctx), token_types.as_in_context(ctx),
            valid_length.as_in_context(ctx))
    toc = time.time()
    log.info('Inference time cost={:.2f} s, Thoughput={:.2f} samples/s'
             .format(toc - tic, len(data_source) / (toc - tic)))

###############################################################################
#                              Export the model                               #
###############################################################################
if __name__ == '__main__':
    evaluate(sample_dataset)
    net.export(os.path.join(args.output_dir, 'static_bert_base_net'), epoch=0)
