# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Script for model comparison between TF and Gluon."""

# pylint: disable=wrong-import-position, wrong-import-order, wildcard-import

import sys
import os
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

parser = argparse.ArgumentParser(description='Comparison script for BERT model in Tensorflow'
                                             'and that in Gluon. This script works with '
                                             'google/bert@f39e881b',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_file', type=str, default='input.txt',
                    help='sample input file for testing')
parser.add_argument('--tf_bert_repo_dir', type=str,
                    default='~/bert/',
                    help='path to the original Tensorflow bert repository. '
                         'The repo should be at f39e881b.')
parser.add_argument('--tf_model_dir', type=str,
                    default='~/uncased_L-12_H-768_A-12/',
                    help='path to the original Tensorflow bert checkpoint directory.')
parser.add_argument('--tf_model_prefix', type=str,
                    default='bert_model.ckpt',
                    help='name of bert checkpoint file.')
parser.add_argument('--tf_config_name', type=str,
                    default='bert_config.json',
                    help='Name of Bert config file')
parser.add_argument('--cased', action='store_true',
                    help='if not set, inputs are converted to lower case')
parser.add_argument('--gluon_dataset', type=str, default='book_corpus_wiki_en_uncased',
                    help='gluon dataset name')
parser.add_argument('--gluon_model', type=str, default='bert_12_768_12',
                    help='gluon model name')
parser.add_argument('--gluon_parameter_file', type=str, default=None,
                    help='gluon parameter file name.')
parser.add_argument('--gluon_vocab_file', type=str, default=None,
                    help='gluon vocab file corresponding to --gluon_parameter_file.')

args = parser.parse_args()

input_file = os.path.expanduser(args.input_file)
tf_bert_repo_dir = os.path.expanduser(args.tf_bert_repo_dir)
tf_model_dir = os.path.expanduser(args.tf_model_dir)
vocab_file = os.path.join(tf_model_dir, 'vocab.txt')
bert_config_file = os.path.join(tf_model_dir, args.tf_config_name)
init_checkpoint = os.path.join(tf_model_dir, args.tf_model_prefix)
do_lower_case = not args.cased
max_length = 128

###############################################################################
#                          Tensorflow MODEL                                   #
###############################################################################
# import tensorflow modules
sys.path.insert(0, tf_bert_repo_dir)

# tensorflow model inference
import modeling
import tokenization
from extract_features import *

# data
num_layers = int(args.gluon_model.split('_')[1])
layer_indexes = list(range(num_layers))
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
examples = read_examples(input_file)

features = convert_examples_to_features(
    examples=examples, seq_length=max_length, tokenizer=tokenizer)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
    master=None,
    tpu_config=tf.contrib.tpu.TPUConfig(
        num_shards=1,
        per_host_input_for_training=is_per_host))
# model
model_fn = model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=init_checkpoint,
    layer_indexes=layer_indexes,
    use_tpu=False,
    use_one_hot_embeddings=False)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    predict_batch_size=1)

input_fn = input_fn_builder(
    features=features, seq_length=max_length)

tensorflow_all_out = []
for result in estimator.predict(input_fn, yield_single_examples=True):
    output_json = collections.OrderedDict()
    tensorflow_all_out_features = []
    all_layers = []
    for (j, layer_index) in enumerate(layer_indexes):
        layer_output = result['layer_output_%d' % j]
        layers = collections.OrderedDict()
        layers['index'] = layer_index
        layers['values'] = layer_output
        all_layers.append(layers)
    tensorflow_out_features = collections.OrderedDict()
    tensorflow_out_features['layers'] = all_layers
    tensorflow_all_out_features.append(tensorflow_out_features)

    output_json['features'] = tensorflow_all_out_features
    tensorflow_all_out.append(output_json)

tf_outputs = [tensorflow_all_out[0]['features'][0]['layers'][t]['values'] for t in layer_indexes]

###############################################################################
#                               Gluon MODEL                                   #
###############################################################################

if args.gluon_parameter_file:
    assert args.gluon_vocab_file, \
        'Must specify --gluon_vocab_file when specifying --gluon_parameter_file'
    with open(args.gluon_vocab_file, 'r') as f:
        vocabulary = nlp.Vocab.from_json(f.read())
    bert, vocabulary = nlp.model.get_model(args.gluon_model,
                                           dataset_name=None,
                                           vocab=vocabulary,
                                           pretrained=not args.gluon_parameter_file,
                                           use_pooler=False,
                                           use_decoder=False,
                                           use_classifier=False)
    try:
        bert.cast('float16')
        bert.load_parameters(args.gluon_parameter_file, ignore_extra=True)
        bert.cast('float32')
    except AssertionError:
        bert.cast('float32')
        bert.load_parameters(args.gluon_parameter_file, ignore_extra=True)
else:
    assert not args.gluon_vocab_file, \
        'Cannot specify --gluon_vocab_file without specifying --gluon_parameter_file'
    bert, vocabulary = nlp.model.get_model(args.gluon_model,
                                           dataset_name=args.gluon_dataset,
                                           pretrained=not args.gluon_parameter_file,
                                           use_pooler=False,
                                           use_decoder=False,
                                           use_classifier=False)

print(bert)
tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=do_lower_case)
dataset = nlp.data.TSVDataset(input_file, field_separator=nlp.data.Splitter(' ||| '))

trans = nlp.data.BERTSentenceTransform(tokenizer, max_length)
dataset = dataset.transform(trans)

bert_dataloader = mx.gluon.data.DataLoader(dataset, batch_size=1,
                                           shuffle=True, last_batch='rollover')

# verify the output of the first sample
for i, seq in enumerate(bert_dataloader):
    input_ids, valid_length, type_ids = seq
    out = bert(input_ids, type_ids,
               valid_length.astype('float32'))
    length = valid_length.asscalar()
    a = tf_outputs[-1][:length]
    b = out[0][:length].asnumpy()

    print('stdev = %s' % (np.std(a - b)))
    mx.test_utils.assert_almost_equal(a, b, atol=5e-6, rtol=5e-6)
    break
