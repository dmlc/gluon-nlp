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
"""Deep Biaffine Parser Model."""

import numpy as np
import mxnet as mx
from mxnet import nd, ndarray, autograd
from mxnet.gluon import nn, loss

from scripts.parsing.common import utils
from gluonnlp.model import apply_weight_drop


class BiaffineParser(nn.Block):
    """A MXNet replicate of biaffine parser, see following paper
    Dozat, T., & Manning, C. D. (2016). Deep biaffine attention for neural dependency parsing.
    arXiv:1611.01734.

    It's a re-implementation of DyNet version
    https://github.com/jcyk/Dynet-Biaffine-dependency-parser

    Parameters
    ----------
    vocab : ParserVocabulary
        built from a data set
    word_dims : int
        word vector dimension
    tag_dims : int
        tag vector dimension
    dropout_dim : int
        keep rate of word dropout (drop out entire embedding)
    lstm_layers : int
        number of lstm layers
    lstm_hiddens : int
        size of lstm hidden states
    dropout_lstm_input : float
        dropout on x in variational RNN
    dropout_lstm_hidden : float
        dropout on h in variational RNN
    mlp_arc_size : int
        output size of MLP for arc feature extraction
    mlp_rel_size : int
        output size of MLP for rel feature extraction
    dropout_mlp : int
        dropout on the output of LSTM
    debug : bool
        debug mode
    """
    def __init__(self, vocab,
                 word_dims,
                 tag_dims,
                 dropout_dim,
                 lstm_layers,
                 lstm_hiddens,
                 dropout_lstm_input,
                 dropout_lstm_hidden,
                 mlp_arc_size,
                 mlp_rel_size,
                 dropout_mlp,
                 debug=False):
        super(BiaffineParser, self).__init__()

        def embedding_from_numpy(_we, trainable=True):
            word_embs = nn.Embedding(_we.shape[0], _we.shape[1],
                                     weight_initializer=mx.init.Constant(_we))
            apply_weight_drop(word_embs, 'weight', dropout_dim, axes=(1,))
            if not trainable:
                word_embs.collect_params().setattr('grad_req', 'null')
            return word_embs

        self._vocab = vocab
        self.word_embs = embedding_from_numpy(vocab.get_word_embs(word_dims))
        self.pret_word_embs = embedding_from_numpy(vocab.get_pret_embs(),
                                                   trainable=False) if vocab.has_pret_embs() \
                              else None
        self.tag_embs = embedding_from_numpy(vocab.get_tag_embs(tag_dims))

        self.f_lstm = nn.Sequential()
        self.b_lstm = nn.Sequential()
        self.f_lstm.add(utils.orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims,
                                                             lstm_hiddens,
                                                             dropout_lstm_hidden, debug))
        self.b_lstm.add(
            utils.orthonormal_VanillaLSTMBuilder(1, word_dims + tag_dims,
                                                 lstm_hiddens,
                                                 dropout_lstm_hidden, debug))
        for _ in range(lstm_layers - 1):
            self.f_lstm.add(
                utils.orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens,
                                                     lstm_hiddens,
                                                     dropout_lstm_hidden, debug))
            self.b_lstm.add(
                utils.orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens,
                                                     lstm_hiddens,
                                                     dropout_lstm_hidden, debug))
        self.dropout_lstm_input = dropout_lstm_input
        self.dropout_lstm_hidden = dropout_lstm_hidden

        mlp_size = mlp_arc_size + mlp_rel_size
        W = utils.orthonormal_initializer(mlp_size, 2 * lstm_hiddens, debug)
        self.mlp_dep_W = self.parameter_from_numpy('mlp_dep_W', W)
        self.mlp_head_W = self.parameter_from_numpy('mlp_head_W', W)
        self.mlp_dep_b = self.parameter_init('mlp_dep_b', (mlp_size,), mx.init.Zero())
        self.mlp_head_b = self.parameter_init('mlp_head_b', (mlp_size,), mx.init.Zero())
        self.mlp_arc_size = mlp_arc_size
        self.mlp_rel_size = mlp_rel_size
        self.dropout_mlp = dropout_mlp

        self.arc_W = self.parameter_init('arc_W', (mlp_arc_size, mlp_arc_size + 1),
                                         init=mx.init.Zero())
        self.rel_W = self.parameter_init('rel_W', (vocab.rel_size * (mlp_rel_size + 1),
                                                   mlp_rel_size + 1),
                                         init=mx.init.Zero())
        self.softmax_loss = loss.SoftmaxCrossEntropyLoss(axis=0, batch_axis=-1)

        self.initialize()

    def parameter_from_numpy(self, name, array):
        """ Create parameter with its value initialized according to a numpy tensor

        Parameters
        ----------
        name : str
            parameter name
        array : np.ndarray
            initiation value

        Returns
        -------
        mxnet.gluon.parameter
            a parameter object
        """
        p = self.params.get(name, shape=array.shape, init=mx.init.Constant(array))
        return p

    def parameter_init(self, name, shape, init):
        """Create parameter given name, shape and initiator

        Parameters
        ----------
        name : str
            parameter name
        shape : tuple
            parameter shape
        init : mxnet.initializer
            an initializer

        Returns
        -------
        mxnet.gluon.parameter
            a parameter object
        """
        p = self.params.get(name, shape=shape, init=init)
        return p

    def forward(self, word_inputs, tag_inputs, arc_targets=None, rel_targets=None):
        # pylint: disable=arguments-differ
        """Run decoding

        Parameters
        ----------
        word_inputs : mxnet.ndarray.NDArray
            word indices of seq_len x batch_size
        tag_inputs : mxnet.ndarray.NDArray
            tag indices of seq_len x batch_size
        arc_targets : mxnet.ndarray.NDArray
            gold arc indices of seq_len x batch_size
        rel_targets : mxnet.ndarray.NDArray
            gold rel indices of seq_len x batch_size
        Returns
        -------
        tuple
            (arc_accuracy, rel_accuracy, overall_accuracy, loss) when training,
            else if given gold target
            then return arc_accuracy, rel_accuracy, overall_accuracy, outputs,
            otherwise return outputs, where outputs is a list of (arcs, rels).
        """
        def flatten_numpy(arr):
            """Flatten nd-array to 1-d column vector

            Parameters
            ----------
            arr : numpy.ndarray
                input tensor

            Returns
            -------
            numpy.ndarray
                A column vector

            """
            return np.reshape(arr, (-1,), 'F')

        is_train = autograd.is_training()
        batch_size = word_inputs.shape[1]
        seq_len = word_inputs.shape[0]
        mask = np.greater(word_inputs, self._vocab.ROOT).astype(np.float32)
        num_tokens = int(np.sum(mask))  # non padding, non root token number

        if is_train or arc_targets is not None:
            mask_1D = flatten_numpy(mask)
            mask_1D_tensor = nd.array(mask_1D)

        unked_words = np.where(word_inputs < self._vocab.words_in_train,
                               word_inputs, self._vocab.UNK)
        word_embs = self.word_embs(nd.array(unked_words, dtype='int'))
        if self.pret_word_embs:
            word_embs = word_embs + self.pret_word_embs(nd.array(word_inputs))
        tag_embs = self.tag_embs(nd.array(tag_inputs))

        # Dropout
        emb_inputs = nd.concat(word_embs, tag_embs, dim=2)  # seq_len x batch_size

        top_recur = utils.biLSTM(self.f_lstm, self.b_lstm, emb_inputs,
                                 dropout_x=self.dropout_lstm_input)
        top_recur = nd.Dropout(data=top_recur, axes=[0], p=self.dropout_mlp)

        W_dep, b_dep = self.mlp_dep_W.data(), self.mlp_dep_b.data()
        W_head, b_head = self.mlp_head_W.data(), self.mlp_head_b.data()
        dep = nd.Dropout(data=utils.leaky_relu(nd.dot(top_recur, W_dep.T) + b_dep),
                         axes=[0], p=self.dropout_mlp)
        head = nd.Dropout(data=utils.leaky_relu(nd.dot(top_recur, W_head.T) + b_head),
                          axes=[0], p=self.dropout_mlp)
        dep, head = nd.transpose(dep, axes=[2, 0, 1]), nd.transpose(head, axes=[2, 0, 1])
        dep_arc, dep_rel = dep[:self.mlp_arc_size], dep[self.mlp_arc_size:]
        head_arc, head_rel = head[:self.mlp_arc_size], head[self.mlp_arc_size:]

        W_arc = self.arc_W.data()
        arc_logits = utils.bilinear(dep_arc, W_arc, head_arc, self.mlp_arc_size,
                                    seq_len, batch_size, num_outputs=1, bias_x=True, bias_y=False)
        # (#head x #dep) x batch_size

        flat_arc_logits = utils.reshape_fortran(arc_logits, (seq_len, seq_len * batch_size))
        # (#head ) x (#dep x batch_size)

        arc_preds = arc_logits.argmax(0)
        # seq_len x batch_size

        if is_train or arc_targets is not None:
            correct = np.equal(arc_preds.asnumpy(), arc_targets)
            arc_correct = correct.astype(np.float32) * mask
            arc_accuracy = np.sum(arc_correct) / num_tokens
            targets_1D = flatten_numpy(arc_targets)
            losses = self.softmax_loss(flat_arc_logits, nd.array(targets_1D))
            arc_loss = nd.sum(losses * mask_1D_tensor) / num_tokens

        if not is_train:
            arc_probs = np.transpose(
                np.reshape(nd.softmax(flat_arc_logits, axis=0).asnumpy(),
                           (seq_len, seq_len, batch_size), 'F'))
        # #batch_size x #dep x #head

        W_rel = self.rel_W.data()
        rel_logits = utils.bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size,
                                    seq_len, batch_size, num_outputs=self._vocab.rel_size,
                                    bias_x=True, bias_y=True)
        # (#head x rel_size x #dep) x batch_size

        flat_rel_logits = utils.reshape_fortran(rel_logits, (seq_len, self._vocab.rel_size,
                                                             seq_len * batch_size))
        # (#head x rel_size) x (#dep x batch_size)

        if is_train: # pylint: disable=using-constant-test
            _target_vec = targets_1D
        else:
            _target_vec = flatten_numpy(arc_preds.asnumpy())
        _target_vec = nd.array(_target_vec).reshape(seq_len * batch_size, 1)
        _target_mat = _target_vec * nd.ones((1, self._vocab.rel_size))

        partial_rel_logits = nd.pick(flat_rel_logits, _target_mat.T, axis=0)
        # (rel_size) x (#dep x batch_size)

        if is_train or arc_targets is not None:
            rel_preds = partial_rel_logits.argmax(0)
            targets_1D = flatten_numpy(rel_targets)
            rel_correct = np.equal(rel_preds.asnumpy(), targets_1D).astype(np.float32) * mask_1D
            rel_accuracy = np.sum(rel_correct) / num_tokens
            losses = self.softmax_loss(partial_rel_logits, nd.array(targets_1D))
            rel_loss = nd.sum(losses * mask_1D_tensor) / num_tokens

        if not is_train:
            rel_probs = np.transpose(np.reshape(nd.softmax(flat_rel_logits.transpose([1, 0, 2]),
                                                           axis=0).asnumpy(),
                                                (self._vocab.rel_size, seq_len,
                                                 seq_len, batch_size), 'F'))
        # batch_size x #dep x #head x #nclasses

        if is_train or arc_targets is not None:
            l = arc_loss + rel_loss
            correct = rel_correct * flatten_numpy(arc_correct)
            overall_accuracy = np.sum(correct) / num_tokens

        if is_train: # pylint: disable=using-constant-test
            return arc_accuracy, rel_accuracy, overall_accuracy, l

        outputs = []

        for msk, arc_prob, rel_prob in zip(np.transpose(mask), arc_probs, rel_probs):
            # parse sentences one by one
            msk[0] = 1.
            sent_len = int(np.sum(msk))
            arc_pred = utils.arc_argmax(arc_prob, sent_len, msk)
            rel_prob = rel_prob[np.arange(len(arc_pred)), arc_pred]
            rel_pred = utils.rel_argmax(rel_prob, sent_len)
            outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len]))

        if arc_targets is not None:
            return arc_accuracy, rel_accuracy, overall_accuracy, outputs
        return outputs

    def save_parameters(self, filename):
        """Save model

        Parameters
        ----------
        filename : str
            path to model file
        """
        params = self._collect_params_with_prefix()
        if self.pret_word_embs:  # don't save word embeddings inside model
            params.pop('pret_word_embs.weight', None)
        arg_dict = {key: val._reduce() for key, val in params.items()}
        ndarray.save(filename, arg_dict)

    def save(self, save_path):
        """Save model

        Parameters
        ----------
        filename : str
            path to model file
        """
        self.save_parameters(save_path)

    def load(self, load_path):
        """Load model

        Parameters
        ----------
        load_path : str
            path to model file
        """
        self.load_parameters(load_path, allow_missing=True)
