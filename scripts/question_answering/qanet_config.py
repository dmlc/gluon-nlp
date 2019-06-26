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

r"""
This file contains all hyperparameters.
"""
import argparse
import mxnet as mx

# Data preprocessing parameters
TRAIN_PARA_LIMIT = 400
TRAIN_QUES_LIMIT = 50
DEV_PARA_LIMIT = 1000
DEV_QUES_LIMIT = 100
ANS_LIMIT = 30
CHAR_LIMIT = 16

GLOVE_FILE_NAME = 'glove.840B.300d'

ACCUM_AVG_TRAIN_CROSS_ENTROPY_PREFIX = 'qanet_accum_avg_train_cross_entropy'
BATCH_TRAIN_CROSS_ENTROPY_PREFIX = 'qanet_batch_train_cross_entropy'
TRAIN_F1 = 'qanet_train_f1.json'
TRAIN_EM = 'qanet_train_em.json'
DEV_CROSS_ENTROPY = 'qanet_dev_cross_entropy.json'
DEV_F1 = 'qanet_dev_f1.json'
DEV_EM = 'qanet_dev_em.json'

EVALUATE_INTERVAL = 0

TRAIN_FLAG = True
CTX = [mx.gpu(0)]

EPOCHS = 60
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16

# model save & load
BEST_MODEL_FILE_NAME = 'qanet_model.params'
BEST_MODEL_EMA_FILE_NAME = 'qanet_ema.params'
NEED_LOAD_TRAINED_MODEL = False
LAST_GLOBAL_STEP = 0

# dropout
LAYERS_DROPOUT = 0.1
p_L = 0.9
WORD_EMBEDDING_DROPOUT = 0.1
CHAR_EMBEDDING_DROPOUT = 0.05
HIGHWAY_LAYERS_DROPOUT = 0.1

# padding parameter
MAX_CONTEXT_SENTENCE_LEN = 400
MAX_QUESTION_SENTENCE_LEN = 50
MAX_CHARACTER_PER_WORD = 16

# embedding parameter
UNK = 1
PAD = 0

DIM_WORD_EMBED = 300
DIM_CHAR_EMBED = 200

NUM_HIGHWAY_LAYERS = 2

# embedding encoder parameter
EMB_ENCODER_CONV_CHANNELS = 128
EMB_ENCODER_CONV_KERNEL_SIZE = 7
EMB_ENCODER_NUM_CONV_LAYERS = 4
EMB_ENCODER_NUM_HEAD = 1
EMB_ENCODER_NUM_BLOCK = 1

# model encoder parameter
MODEL_ENCODER_CONV_KERNEL_SIZE = 5
MODEL_ENCODER_CONV_CHANNELS = 128
MODEL_ENCODER_NUM_CONV_LAYERS = 2
MODEL_ENCODER_NUM_HEAD = 1
MODEL_ENCODER_NUM_BLOCK = 7

# opt parameter
INIT_LEARNING_RATE = 0.001
CLIP_GRADIENT = 5
WEIGHT_DECAY = 3e-7
BETA1 = 0.8
BETA2 = 0.999
EPSILON = 1e-7
EXPONENTIAL_MOVING_AVERAGE_DECAY = 0.9999
WARM_UP_STEPS = 1000

# evaluate
MAX_ANSWER_LENS = 30


def get_args():
    """Get console arguments
    """
    parser = argparse.ArgumentParser(description='Question Answering example using QANet & SQuAD',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='qanet_output',
                        help='directory path to save the final model and training log')
    parser.add_argument('--save_cross_entropy', default=False, action='store_true',
                        help='Store cross entropy results after each epoch')

    options = parser.parse_args()
    return options
