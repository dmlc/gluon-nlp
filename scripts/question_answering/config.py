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
This file contains all hyperparameters.
"""
import argparse

import mxnet as mx

parser = argparse.ArgumentParser(
    description='The hyperparameters for QANet and optimizer.')

# configure
parser.add_argument('--use_gpu', type=bool, default=True,
                    help='Whether use gpu for train and eval.')
parser.add_argument('--gpu_list', type=str,
                    default='1', help='List of gpu index.')
parser.add_argument('--prefix_model', type=str, default='model_',
                    help='The prefix name for model saving.')
parser.add_argument('--prefix_trainer', type=str, default='trainer_',
                    help='The prefix name for trainer saving.')
parser.add_argument('--load_trained_model', type=bool,
                    default=False, help='Whether load trained model.')
parser.add_argument('--trained_model_name', type=str,
                    default='', help='File name of trained model.')
parser.add_argument('--trained_trainer_name', type=str,
                    default='', help='File name of trained trainer.')
parser.add_argument('--evaluate_interval', type=int, default=5000,
                    help='Evaluate the model after how many batchs.')
parser.add_argument('--is_train', type=bool, default=True,
                    help='Whether train or evaluate the model.')
parser.add_argument('--data_path', type=str,
                    default='../data/', help='The root dir of data.')
parser.add_argument('--train_file_name', type=str,
                    default='train-v1.1.json', help='The train file name.')
parser.add_argument('--dev_file_name', type=str,
                    default='dev-v1.1.json', help='The dev file name.')
parser.add_argument('--processed_train_file_name', type=str,
                    default='train_sorted.json', help='The processed train file name.')
parser.add_argument('--processed_dev_file_name', type=str,
                    default='dev_sorted.json', help='The processed dev file name.')
parser.add_argument('--glove_file_name', type=str,
                    default='glove.840B.300d.txt', help='The glove file name.')
parser.add_argument('--word_emb_file_name', type=str,
                    default='word_emb.json', help='The name of word embedding file.')
parser.add_argument('--char_emb_file_name', type=str,
                    default='char_emb.json', help='The name of char embedding file.')
parser.add_argument('--train_batch_size', type=int, default=32,
                    help='Train batch size, default is 32.')
parser.add_argument('--eval_batch_size', type=int, default=16,
                    help='Evaluate batch size, default is 16.')
parser.add_argument('--epochs', type=int, default=60, help='The train epochs.')
parser.add_argument('--layers_dropout', type=float, default=0.1,
                    help='The dropout rate between two layers, default is 0.1.')
parser.add_argument('--p_l', type=float, default=0.9,
                    help='The init keep probability of stochastic dropout layer, default is 0.9.')
parser.add_argument('--word_emb_dropout', type=float, default=0.1,
                    help='The dropout probability of word embedding matrix, default is 0.1 .')
parser.add_argument('--char_emb_dropout', type=float, default=0.05,
                    help='The dropout probability of char embedding matrix, default is 0.05 .')
parser.add_argument('--max_context_sentence_len', type=int, default=400,
                    help='The limit lens of context sentence, default is 400.')
parser.add_argument('--max_question_sentence_len', type=int, default=50,
                    help='The limit lens of question sentence, default is 50.')
parser.add_argument('--max_character_per_word', type=int, default=16,
                    help='The limit lens of characters per word, default is 16.')
parser.add_argument('--UNK', type=int, default=1,
                    help='The idx of OOV flag, default is 1.')
parser.add_argument('--PAD', type=int, default=0,
                    help='The idx of padding flag, default is 0.')

# hyperparameters for QANet
parser.add_argument('--word_emb_dim', type=int, default=300,
                    help='The dimensions of word embedding, default is 300.')
parser.add_argument('--char_emb_dim', type=int, default=200,
                    help='The dimensions of word embedding, default is 200.')
parser.add_argument('--character_corpus', type=int, default=1372+2,
                    help='The size of character corpus, default is 1374. ')
parser.add_argument('--word_corpus', type=int, default=86822+2,
                    help='The size of word corpus, default is 86824. ')
parser.add_argument('--highway_layers', type=int, default=2,
                    help='The layers of highway, default is 2.')
parser.add_argument('--char_conv_filters', type=tuple, default=(50, 75, 100),
                    help='default set of char conv filters\' size')
parser.add_argument('--char_conv_ngrams', type=tuple, default=(2, 3, 4),
                    help='default set of char conv filters\' size')


parser.add_argument('--emb_encoder_conv_channels', type=int, default=128,
                    help='The hidden size of embedding encoder, default is 128.')
parser.add_argument('--emb_encoder_conv_kernerl_size', type=int, default=7,
                    help='The conv kernerl size in embedding encoder, default is 7.')
parser.add_argument('--emb_encoder_num_conv_layers',
                    type=int, default=4, help='The number of conv layers, default is 4.')
parser.add_argument('--emb_encoder_num_head', type=int, default=2,
                    help='The number of heads in self-attention, default is 2.')
parser.add_argument('--emb_encoder_num_block', type=int, default=1,
                    help='The number of embedding encoder, default is 1.')

parser.add_argument('--model_encoder_conv_channels', type=int, default=128,
                    help='The hidden size of model encoder, default is 128.')
parser.add_argument('--model_encoder_conv_kernel_size', type=int, default=5,
                    help='The conv kernerl size in embedding encoder, default is 5.')
parser.add_argument('--model_encoder_conv_layers',
                    type=int, default=2, help='The number of conv layers, default is 2.')
parser.add_argument('--model_encoder_num_head', type=int, default=2,
                    help='The number of heads in self-attention, default is 2.')
parser.add_argument('--model_encoder_num_block', type=int, default=7,
                    help='The number of embedding encoder, default is 7.')

# opt parameter
parser.add_argument('--init_learning_rate', type=float, default=0.001,
                    help='The init learning rate, default is 0.001 .')
parser.add_argument('--clip_gradient', type=int, default=5,
                    help='The clip value of gradient, default is 5.')
parser.add_argument('--weight_decay', type=float, default=3e-7,
                    help='The l2 weight decay, default is 3e-7.')
parser.add_argument('--beta1', type=float, default=0.8,
                    help='The beta1 in adam optimizer, default is 0.8 .')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='The beta2 in adam optimizer, default is 0.999 .')
parser.add_argument('--epsilon', type=float, default=1e-7,
                    help='The epsilon in adam optimizer, default is 1e-7.')
parser.add_argument('--ema_decay', type=float, default=0.9999,
                    help='The exponential moving average decay rate, default is 0.9999 .')
parser.add_argument('--warm_up_steps', type=int, default=1000,
                    help='The warm up steps of learning rate, default is 1000.')

parser.add_argument('--max_answer_len', type=int, default=30,
                    help='The constraint of answer lens, default is 30.')

opt = parser.parse_args()
print(opt)

CTX = []
if opt.use_gpu:
    for idx in opt.gpu_list.split(','):
        CTX.append(mx.gpu(int(idx)))
else:
    CTX.append(mx.cpu())
