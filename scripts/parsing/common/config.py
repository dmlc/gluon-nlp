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
"""Training config."""

import os
import pickle

from scripts.parsing.common.savable import Savable


class _Config(Savable):
    def __init__(self, train_file, dev_file, test_file, save_dir,
                 pretrained_embeddings_file=None, min_occur_count=2,
                 lstm_layers=3, word_dims=100, tag_dims=100, dropout_emb=0.33, lstm_hiddens=400,
                 dropout_lstm_input=0.33,
                 dropout_lstm_hidden=0.33, mlp_arc_size=500, mlp_rel_size=100,
                 dropout_mlp=0.33, learning_rate=2e-3, decay=.75, decay_steps=5000,
                 beta_1=.9, beta_2=.9, epsilon=1e-12,
                 num_buckets_train=40,
                 num_buckets_valid=10, num_buckets_test=10,
                 train_iters=50000, train_batch_size=5000, debug=False):
        """Internal structure for hyper parameters, intended for pickle serialization.

        May be replaced by a dict, but this class provides intuitive properties
        and saving/loading mechanism

        Parameters
        ----------
        train_file
        dev_file
        test_file
        save_dir
        pretrained_embeddings_file
        min_occur_count
        lstm_layers
        word_dims
        tag_dims
        dropout_emb
        lstm_hiddens
        dropout_lstm_input
        dropout_lstm_hidden
        mlp_arc_size
        mlp_rel_size
        dropout_mlp
        learning_rate
        decay
        decay_steps
        beta_1
        beta_2
        epsilon
        num_buckets_train
        num_buckets_valid
        num_buckets_test
        train_iters
        train_batch_size
        debug
        """
        super(_Config, self).__init__()
        self.pretrained_embeddings_file = pretrained_embeddings_file
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.min_occur_count = min_occur_count
        self.save_dir = save_dir
        self.lstm_layers = lstm_layers
        self.word_dims = word_dims
        self.tag_dims = tag_dims
        self.dropout_emb = dropout_emb
        self.lstm_hiddens = lstm_hiddens
        self.dropout_lstm_input = dropout_lstm_input
        self.dropout_lstm_hidden = dropout_lstm_hidden
        self.mlp_arc_size = mlp_arc_size
        self.mlp_rel_size = mlp_rel_size
        self.dropout_mlp = dropout_mlp
        self.learning_rate = learning_rate
        self.decay = decay
        self.decay_steps = decay_steps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.num_buckets_train = num_buckets_train
        self.num_buckets_valid = num_buckets_valid
        self.num_buckets_test = num_buckets_test
        self.train_iters = train_iters
        self.train_batch_size = train_batch_size
        self.debug = debug

    @property
    def save_model_path(self):
        return os.path.join(self.save_dir, 'model.bin')

    @property
    def save_vocab_path(self):
        return os.path.join(self.save_dir, 'vocab.pkl')

    @property
    def save_config_path(self):
        return os.path.join(self.save_dir, 'config.pkl')

    def save(self, path=None):
        if not path:
            path = self.save_config_path
        with open(path, 'wb') as f:
            pickle.dump(self, f)
