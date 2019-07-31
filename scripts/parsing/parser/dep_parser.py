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
"""Deep Biaffine Dependency Parser driver class and script."""

import math
import os
import numpy as np

import mxnet as mx
from mxnet import gluon, autograd

from scripts.parsing.common.config import _Config
from scripts.parsing.common.data import ParserVocabulary, DataLoader, ConllWord, ConllSentence
from scripts.parsing.common.exponential_scheduler import ExponentialScheduler
from scripts.parsing.common.utils import init_logger, mxnet_prefer_gpu, Progbar
from scripts.parsing.parser.biaffine_parser import BiaffineParser
from scripts.parsing.parser.evaluate import evaluate_official_script


class DepParser:
    """User interfaces for biaffine dependency parser.

    It wraps a biaffine model inside, provides training, evaluating and parsing.
    """

    def __init__(self):
        super().__init__()
        self._parser = None
        self._vocab = None

    def train(self, train_file, dev_file, test_file, save_dir,
              pretrained_embeddings=None, min_occur_count=2,
              lstm_layers=3, word_dims=100, tag_dims=100, dropout_emb=0.33, lstm_hiddens=400,
              dropout_lstm_input=0.33, dropout_lstm_hidden=0.33,
              mlp_arc_size=500, mlp_rel_size=100,
              dropout_mlp=0.33, learning_rate=2e-3, decay=.75, decay_steps=5000,
              beta_1=.9, beta_2=.9, epsilon=1e-12,
              num_buckets_train=40,
              num_buckets_valid=10, num_buckets_test=10, train_iters=50000, train_batch_size=5000,
              test_batch_size=5000, validate_every=100, save_after=5000, debug=False):
        """Train a deep biaffine dependency parser.

        Parameters
        ----------
        train_file : str
            path to training set
        dev_file : str
            path to dev set
        test_file : str
            path to test set
        save_dir : str
            a directory for saving model and related meta-data
        pretrained_embeddings : tuple
            (embedding_name, source), used for gluonnlp.embedding.create(embedding_name, source)
        min_occur_count : int
            threshold of rare words, which will be replaced with UNKs,
        lstm_layers : int
            layers of lstm
        word_dims : int
            dimension of word embedding
        tag_dims : int
            dimension of tag embedding
        dropout_emb : float
            word dropout
        lstm_hiddens : int
            size of lstm hidden states
        dropout_lstm_input : int
            dropout on x in variational RNN
        dropout_lstm_hidden : int
            dropout on h in variational RNN
        mlp_arc_size : int
            output size of MLP for arc feature extraction
        mlp_rel_size : int
            output size of MLP for rel feature extraction
        dropout_mlp : float
            dropout on the output of LSTM
        learning_rate : float
            learning rate
        decay : float
            see ExponentialScheduler
        decay_steps : int
            see ExponentialScheduler
        beta_1 : float
            see ExponentialScheduler
        beta_2 : float
            see ExponentialScheduler
        epsilon : float
            see ExponentialScheduler
        num_buckets_train : int
            number of buckets for training data set
        num_buckets_valid : int
            number of buckets for dev data set
        num_buckets_test : int
            number of buckets for testing data set
        train_iters : int
            training iterations
        train_batch_size : int
            training batch size
        test_batch_size : int
            test batch size
        validate_every : int
            validate on dev set every such number of batches
        save_after : int
            skip saving model in early epochs
        debug : bool
            debug mode

        Returns
        -------
        DepParser
            parser itself
        """
        logger = init_logger(save_dir)
        config = _Config(train_file, dev_file, test_file, save_dir, pretrained_embeddings,
                         min_occur_count,
                         lstm_layers, word_dims, tag_dims, dropout_emb, lstm_hiddens,
                         dropout_lstm_input, dropout_lstm_hidden, mlp_arc_size, mlp_rel_size,
                         dropout_mlp, learning_rate, decay, decay_steps,
                         beta_1, beta_2, epsilon, num_buckets_train, num_buckets_valid,
                         num_buckets_test, train_iters,
                         train_batch_size, debug)
        config.save()
        self._vocab = vocab = ParserVocabulary(train_file,
                                               pretrained_embeddings,
                                               min_occur_count)
        vocab.save(config.save_vocab_path)
        vocab.log_info(logger)

        with mx.Context(mxnet_prefer_gpu()):
            self._parser = parser = BiaffineParser(vocab, word_dims, tag_dims,
                                                   dropout_emb,
                                                   lstm_layers,
                                                   lstm_hiddens, dropout_lstm_input,
                                                   dropout_lstm_hidden,
                                                   mlp_arc_size,
                                                   mlp_rel_size, dropout_mlp, debug)
            parser.initialize()
            scheduler = ExponentialScheduler(learning_rate, decay, decay_steps)
            optimizer = mx.optimizer.Adam(learning_rate, beta_1, beta_2, epsilon,
                                          lr_scheduler=scheduler)
            trainer = gluon.Trainer(parser.collect_params(), optimizer=optimizer)
            data_loader = DataLoader(train_file, num_buckets_train, vocab)
            global_step = 0
            best_UAS = 0.
            batch_id = 0
            epoch = 1
            total_epoch = math.ceil(train_iters / validate_every)
            logger.info('Epoch %d out of %d', epoch, total_epoch)
            bar = Progbar(target=min(validate_every, data_loader.samples))
            while global_step < train_iters:
                for words, tags, arcs, rels in data_loader.get_batches(batch_size=train_batch_size,
                                                                       shuffle=True):
                    with autograd.record():
                        arc_accuracy, _, _, loss = parser.forward(words, tags, arcs, rels)
                        loss_value = loss.asscalar()
                    loss.backward()
                    trainer.step(train_batch_size)
                    batch_id += 1
                    try:
                        bar.update(batch_id,
                                   exact=[('UAS', arc_accuracy, 2),
                                          ('loss', loss_value)])
                    except OverflowError:
                        pass  # sometimes loss can be 0 or infinity, crashes the bar

                    global_step += 1
                    if global_step % validate_every == 0:
                        bar = Progbar(target=min(validate_every, train_iters - global_step))
                        batch_id = 0
                        UAS, LAS, speed = evaluate_official_script(parser, vocab,
                                                                   num_buckets_valid,
                                                                   test_batch_size,
                                                                   dev_file,
                                                                   os.path.join(save_dir,
                                                                                'valid_tmp'))
                        logger.info('Dev: UAS %.2f%% LAS %.2f%% %d sents/s', UAS, LAS, speed)
                        epoch += 1
                        if global_step < train_iters:
                            logger.info('Epoch %d out of %d', epoch, total_epoch)
                        if global_step > save_after and UAS > best_UAS:
                            logger.info('- new best score!')
                            best_UAS = UAS
                            parser.save(config.save_model_path)

        # When validate_every is too big
        if not os.path.isfile(config.save_model_path) or best_UAS != UAS:
            parser.save(config.save_model_path)

        return self

    def load(self, path):
        """Load from disk

        Parameters
        ----------
        path : str
            path to the directory which typically contains a config.pkl file and a model.bin file

        Returns
        -------
        DepParser
            parser itself
        """
        config = _Config.load(os.path.join(path, 'config.pkl'))
        config.save_dir = path  # redirect root path to what user specified
        self._vocab = vocab = ParserVocabulary.load(config.save_vocab_path)
        with mx.Context(mxnet_prefer_gpu()):
            self._parser = BiaffineParser(vocab, config.word_dims, config.tag_dims,
                                          config.dropout_emb,
                                          config.lstm_layers,
                                          config.lstm_hiddens, config.dropout_lstm_input,
                                          config.dropout_lstm_hidden,
                                          config.mlp_arc_size, config.mlp_rel_size,
                                          config.dropout_mlp, config.debug)
            self._parser.load(config.save_model_path)
        return self

    def evaluate(self, test_file, save_dir=None, logger=None,
                 num_buckets_test=10, test_batch_size=5000):
        """Run evaluation on test set

        Parameters
        ----------
        test_file : str
            path to test set
        save_dir : str
            where to store intermediate results and log
        logger : logging.logger
            logger for printing results
        num_buckets_test : int
            number of clusters for sentences from test set
        test_batch_size : int
            batch size of test set

        Returns
        -------
        tuple
            UAS, LAS
        """
        parser = self._parser
        vocab = self._vocab
        with mx.Context(mxnet_prefer_gpu()):
            UAS, LAS, speed = evaluate_official_script(parser, vocab, num_buckets_test,
                                                       test_batch_size, test_file,
                                                       os.path.join(save_dir, 'valid_tmp'))
        if logger is None:
            logger = init_logger(save_dir, 'test.log')
        logger.info('Test: UAS %.2f%% LAS %.2f%% %d sents/s', UAS, LAS, speed)

        return UAS, LAS

    def parse(self, sentence):
        """Parse raw sentence into ConllSentence

        Parameters
        ----------
        sentence : list
            a list of (word, tag) tuples

        Returns
        -------
        ConllSentence
            ConllSentence object
        """
        words = np.zeros((len(sentence) + 1, 1), np.int32)
        tags = np.zeros((len(sentence) + 1, 1), np.int32)
        words[0, 0] = ParserVocabulary.ROOT
        tags[0, 0] = ParserVocabulary.ROOT
        vocab = self._vocab

        for i, (word, tag) in enumerate(sentence):
            words[i + 1, 0], tags[i + 1, 0] = vocab.word2id(word.lower()), vocab.tag2id(tag)

        with mx.Context(mxnet_prefer_gpu()):
            outputs = self._parser.forward(words, tags)
        words = []
        for arc, rel, (word, tag) in zip(outputs[0][0], outputs[0][1], sentence):
            words.append(ConllWord(idx=len(words) + 1, form=word, pos=tag,
                                   head=arc, relation=vocab.id2rel(rel)))
        return ConllSentence(words)


if __name__ == '__main__':
    dep_parser = DepParser()
    dep_parser.train(train_file='tests/data/biaffine/ptb/train.conllx',
                     dev_file='tests/data/biaffine/ptb/dev.conllx',
                     test_file='tests/data/biaffine/ptb/test.conllx',
                     save_dir='tests/data/biaffine/model',
                     pretrained_embeddings=('glove', 'glove.6B.100d'))
    dep_parser.load('tests/data/biaffine/model')
    dep_parser.evaluate(test_file='tests/data/biaffine/ptb/test.conllx',
                        save_dir='tests/data/biaffine/model')

    sent = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'),
            ('of', 'IN'), ('chamber', 'NN'), ('music', 'NN'), ('?', '.')]
    print(dep_parser.parse(sent))
