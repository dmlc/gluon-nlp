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

import os
import sys
import pickle

sample_input_file = "/home/ubuntu/pytorch-pretrained-BERT/samples/input.txt"
tf_bert_dir = "/home/ubuntu/bert/"
tf_model_dir = "/home/ubuntu/bert/uncased_L-12_H-768_A-12/"
vocab_file = tf_model_dir + "vocab.txt"
bert_config_file = tf_model_dir + "bert_config.json"
init_checkpoint = tf_model_dir + "bert_model.ckpt"
do_lower_case = True
max_length = 128

###############################################################################
#                          Tensorflow MODEL                                   #
###############################################################################
# import tensorflow modules
sys.path.insert(0, tf_bert_dir)

# tensorflow model inference
import modeling
import tokenization
from extract_features import *

# data
layer_indexes = list(range(12))
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
examples = read_examples(sample_input_file)

features = convert_examples_to_features(
    examples=examples, seq_length=max_length, tokenizer=tokenizer)

unique_id_to_feature = {}
for feature in features:
    unique_id_to_feature[feature.unique_id] = feature

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
    unique_id = int(result["unique_id"])
    feature = unique_id_to_feature[unique_id]
    output_json = collections.OrderedDict()
    output_json["linex_index"] = unique_id
    tensorflow_all_out_features = []
    all_layers = []
    for (j, layer_index) in enumerate(layer_indexes):
        layer_output = result["layer_output_%d" % j]
        layers = collections.OrderedDict()
        layers["index"] = layer_index
        layers["values"] = layer_output
        all_layers.append(layers)
    tensorflow_out_features = collections.OrderedDict()
    tensorflow_out_features["layers"] = all_layers
    tensorflow_all_out_features.append(tensorflow_out_features)

    output_json["features"] = tensorflow_all_out_features
    tensorflow_all_out.append(output_json)

tf_outputs = list(tensorflow_all_out[0]['features'][0]['layers'][t]['values'] for t in layer_indexes)

###############################################################################
#                               Gluon MODEL                                   #
###############################################################################
# import gluon modules
import gluonnlp as nlp
import numpy as np
import mxnet as mx
from mxnet import gluon
from gluonnlp.model import bert_12_768_12
from tokenizer import FullTokenizer
from gluonnlp.data import TSVDataset
from dataset import MRPCDataset, ClassificationTransform, BERTTransform

ctx = mx.cpu()
bert, vocabulary = bert_12_768_12(dataset_name='book_corpus_wiki_en_uncased',
                                  pretrained=True, ctx=ctx, use_pooler=False,
                                  use_decoder=False, use_classifier=False)
tokenizer = FullTokenizer(vocabulary, do_lower_case=True)
dataset = TSVDataset(sample_input_file, field_separator=nlp.data.Splitter(' ||| '))

print(dataset[0])
trans = BERTTransform(tokenizer, max_length)
dataset = dataset.transform(trans)

bert_dataloader = mx.gluon.data.DataLoader(dataset, batch_size=1,
                                           shuffle=True, last_batch='rollover')
for i, seq in enumerate(bert_dataloader):
    input_ids, valid_length, type_ids = seq
    out = bert(input_ids, type_ids,
               valid_length.astype('float32'))
    length = valid_length.asscalar()
    a = tf_outputs[-1][:length]
    b = out[0][:length].asnumpy()
    
    print('stdev = ', np.std(a -b))
    mx.test_utils.assert_almost_equal(a, b, atol=1e-4, rtol=1e-4)
    mx.test_utils.assert_almost_equal(a, b, atol=1e-5, rtol=1e-5)
    mx.test_utils.assert_almost_equal(a, b, atol=5e-6, rtol=5e-6)
