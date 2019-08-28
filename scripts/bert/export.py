"""
Export the BERT Model for Deployment

====================================

This script exports the BERT model to a hybrid model serialized as a symbol.json file,
which is suitable for deployment, or use with MXNet Module API.

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
from gluonnlp.model import get_model
from model.classification import BERTClassifier, BERTRegression
from model.qa import BertForQA

parser = argparse.ArgumentParser(description='Export hybrid BERT base model.')

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
                    choices=['classification', 'regression', 'question_answering'],
                    required=True,
                    help='Task to export. Options are "classification", "regression", '
                         '"question_answering"')

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

parser.add_argument('--dropout',
                    type=float,
                    default=0.1,
                    help='The dropout probability for the classification/regression head.')

args = parser.parse_args()

# create output dir
output_dir = args.output_dir
nlp.utils.mkdir(output_dir)

###############################################################################
#                                Logging                                      #
###############################################################################

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s',
                              datefmt='%H:%M:%S')
fh = logging.FileHandler(os.path.join(args.output_dir, 'hybrid_export_bert.log'), mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)
log.info(args)

###############################################################################
#                              Hybridize the model                            #
###############################################################################

seq_length = args.seq_length

if args.task == 'classification':
    bert, _ = get_model(
        name=args.model_name,
        dataset_name=args.dataset_name,
        pretrained=False,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False,
        seq_length=args.seq_length)
    net = BERTClassifier(bert, num_classes=2, dropout=args.dropout)
elif args.task == 'regression':
    bert, _ = get_model(
        name=args.model_name,
        dataset_name=args.dataset_name,
        pretrained=False,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False,
        seq_length=args.seq_length)
    net = BERTRegression(bert, dropout=args.dropout)
elif args.task == 'question_answering':
    bert, _ = get_model(
        name=args.model_name,
        dataset_name=args.dataset_name,
        pretrained=False,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False,
        seq_length=args.seq_length)
    net = BertForQA(bert)
else:
    raise ValueError('unknown task: %s'%args.task)

if args.model_parameters:
    net.load_parameters(args.model_parameters)
else:
    net.initialize()
    warnings.warn('--model_parameters is not provided. The parameter checkpoint (.params) '
                  'file will be created based on default parameter initialization.')

net.hybridize(static_alloc=True, static_shape=True)

###############################################################################
#                            Prepare dummy input data                         #
###############################################################################

test_batch_size = 1

inputs = mx.nd.arange(test_batch_size * seq_length)
inputs = inputs.reshape(shape=(test_batch_size, seq_length))
token_types = mx.nd.zeros_like(inputs)
valid_length = mx.nd.arange(test_batch_size)
batch = inputs, token_types, valid_length

def export(batch, prefix):
    """Export the model."""
    log.info('Exporting the model ... ')
    inputs, token_types, valid_length = batch
    net(inputs, token_types, valid_length)
    net.export(prefix, epoch=0)
    assert os.path.isfile(prefix + '-symbol.json')
    assert os.path.isfile(prefix + '-0000.params')

def infer(batch, prefix):
    """Evaluate the model on a mini-batch."""
    log.info('Start inference ... ')

    # import with SymbolBlock. Alternatively, you can use Module.load APIs.
    imported_net = mx.gluon.nn.SymbolBlock.imports(prefix + '-symbol.json',
                                                   ['data0', 'data1', 'data2'],
                                                   prefix + '-0000.params')
    tic = time.time()
    # run forward inference
    inputs, token_types, valid_length = batch
    num_trials = 10
    for _ in range(num_trials):
        imported_net(inputs, token_types, valid_length)
    mx.nd.waitall()
    toc = time.time()
    log.info('Inference time cost={:.2f} s, Thoughput={:.2f} samples/s'
             .format(toc - tic, num_trials / (toc - tic)))


###############################################################################
#                              Export the model                               #
###############################################################################
if __name__ == '__main__':
    prefix = os.path.join(args.output_dir, args.task)
    export(batch, prefix)
    infer(batch, prefix)
