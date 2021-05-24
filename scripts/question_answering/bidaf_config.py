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

"""Configuration for running BiDAF model"""

import argparse


def get_args():
    """Get console arguments
    """
    parser = argparse.ArgumentParser(description='Question Answering example using BiDAF & SQuAD',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', default=False, action='store_true',
                        help='Run training')
    parser.add_argument('--evaluate', default=False, action='store_true',
                        help='Run evaluation on dev dataset')
    parser.add_argument('--epochs', type=int, default=12, help='Upper epoch limit')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='Dimension of the word embedding')
    parser.add_argument('--embedding_file_name', type=str, default='glove.6B.100d',
                        help='Embedding file name')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--ctx_embedding_num_layers', type=int, default=2,
                        help='Number of layers in Contextual embedding layer of BiDAF')
    parser.add_argument('--highway_num_layers', type=int, default=2,
                        help='Number of layers in Highway layer of BiDAF')
    parser.add_argument('--modeling_num_layers', type=int, default=2,
                        help='Number of layers in Modeling layer of BiDAF')
    parser.add_argument('--output_num_layers', type=int, default=1,
                        help='Number of layers in Output layer of BiDAF')
    parser.add_argument('--batch_size', type=int, default=60, help='Batch size')
    parser.add_argument('--ctx_max_len', type=int, default=400, help='Maximum length of a context')
    parser.add_argument('--q_max_len', type=int, default=30, help='Maximum length of a question')
    parser.add_argument('--word_max_len', type=int, default=16, help='Maximum characters in a word')
    parser.add_argument('--answer_max_len', type=int, default=30, help='Maximum tokens in answer')
    parser.add_argument('--optimizer', type=str, default='adadelta', help='optimization algorithm')
    parser.add_argument('--lr', type=float, default=0.5, help='Initial learning rate')
    parser.add_argument('--rho', type=float, default=0.9,
                        help='Adadelta decay rate for both squared gradients and delta.')
    parser.add_argument('--lr_warmup_steps', type=int, default=0,
                        help='Defines how many iterations to spend on warming up learning rate')
    parser.add_argument('--clip', type=float, default=0, help='gradient clipping')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for parameter updates')
    parser.add_argument('--log_interval', type=int, default=0, metavar='N',
                        help='Report interval applied to last epoch only')
    parser.add_argument('--early_stop', type=int, default=9,
                        help='Apply early stopping for the last epoch. Stop after # of consequent '
                             '# of times F1 is lower than max. Should be used with log_interval')
    parser.add_argument('--terminate_training_on_reaching_F1_threshold', type=float, default=0,
                        help='Some tasks, like DAWNBenchmark requires to minimize training time '
                             'while reaching a particular F1 metric. This parameter controls if '
                             'training should be terminated as soon as F1 is reached to minimize '
                             'training time and cost. It would force to do evaluation every epoch.')
    parser.add_argument('--save_dir', type=str, default='bidaf_output',
                        help='directory path to save the final model and training log')
    parser.add_argument('--word_vocab_path', type=str, default=None,
                        help='Path to preprocessed word-level vocabulary')
    parser.add_argument('--char_vocab_path', type=str, default=None,
                        help='Path to preprocessed character-level vocabulary')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Coma-separated ids of the gpu to use. Empty means to use cpu.')
    parser.add_argument('--train_unk_token', default=False, action='store_true',
                        help='Should train unknown token of embedding')
    parser.add_argument('--filter_train_examples', default=True, action='store_false',
                        help='Filter contexts if the answer is after ctx_max_len')
    parser.add_argument('--save_prediction_path', type=str, default='',
                        help='Path to save predictions')
    parser.add_argument('--use_exponential_moving_average', default=True, action='store_false',
                        help='Should averaged copy of parameters been stored and used '
                             'during evaluation.')
    parser.add_argument('--exponential_moving_average_weight_decay', type=float, default=0.999,
                        help='Weight decay used in exponential moving average')

    options = parser.parse_args()
    return options
