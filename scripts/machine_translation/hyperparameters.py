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
"""Hyperparameters for transformer, for past reference only."""

# parameters for dataset
src_lang = 'en'
tgt_lang = 'de'
src_max_len = -1
tgt_max_len = -1

# parameters for model
num_units = 512
hidden_size = 2048
dropout = 0.1
epsilon = 0.1
num_layers = 6
num_heads = 8
scaled = True

# parameters for training
optimizer = 'adam'
epochs = 3
batch_size = 2700
test_batch_size = 256
num_accumulated = 1
lr = 2
warmup_steps = 1
save_dir = 'transformer_en_de_u512'
average_start = 1
num_buckets = 20
log_interval = 10
bleu = '13a'

#parameters for testing
beam_size = 4
lp_alpha = 0.6
lp_k = 5
