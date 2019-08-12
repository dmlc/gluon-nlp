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
"""Utility classes."""

import logging
import math
import os
import sys
import time

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import rnn, contrib

from .data import ParserVocabulary
from .tarjan import Tarjan


class Progbar:
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update

    Parameters
    ----------
    target : int
        Total number of steps expected.
    width : int
        Progress bar width.
    verbose : int
        Verbosity level. Options are 1 and 2.
    """
    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None, strict=None):
        """
        Updates the progress bar.

        Parameters
        ----------
        current : int
            Index of current step.
        values : List of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        exact : List of tuples (name, value_for_last_step).
            The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []
        strict = strict or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)

        for cells in exact:
            k, v, w = cells[0], cells[1], 4
            if len(cells) == 3:
                w = cells[2]
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1, w]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')

            numdigits = 0 if self.target == 0 or math.isnan(self.target) \
                        else int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = 0 if self.target == 0 else float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += (' - %s: %.' + str(self.sum_values[k][2]) + 'f') % (
                        k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * ' ')

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write('\n')

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + '\n')

    def add(self, n, values=None):
        values = values or []
        self.update(self.seen_so_far + n, values)


def mxnet_prefer_gpu():
    """If gpu available return gpu, else cpu

    Returns
    -------
    context : Context
        The preferable GPU context.
    """
    gpu = int(os.environ.get('MXNET_GPU', default=0))
    if gpu in mx.test_utils.list_gpus():
        return mx.gpu(gpu)
    return mx.cpu()


def init_logger(root_dir, name='train.log'):
    """Initialize a logger

    Parameters
    ----------
    root_dir : str
        directory for saving log
    name : str
        name of logger

    Returns
    -------
    logger : logging.Logger
        a logger
    """
    os.makedirs(root_dir, exist_ok=True)
    log_formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler('{0}/{1}'.format(root_dir, name), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens,
                                   dropout_h=0., debug=False):
    """Build a standard LSTM cell, with variational dropout,
    with weights initialized to be orthonormal (https://arxiv.org/abs/1312.6120)

    Parameters
    ----------
    lstm_layers : int
        Currently only support one layer
    input_dims : int
        word vector dimensions
    lstm_hiddens : int
        hidden size
    dropout_h : float
        dropout on hidden states
    debug : bool
        set to True to skip orthonormal initialization

    Returns
    -------
    lstm_cell : VariationalDropoutCell
        A LSTM cell
    """
    assert lstm_layers == 1, 'only accept one layer lstm'
    W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + input_dims, debug)
    W_h, W_x = W[:, :lstm_hiddens], W[:, lstm_hiddens:]
    b = nd.zeros((4 * lstm_hiddens,))
    b[lstm_hiddens:2 * lstm_hiddens] = -1.0
    lstm_cell = rnn.LSTMCell(input_size=input_dims, hidden_size=lstm_hiddens,
                             i2h_weight_initializer=mx.init.Constant(np.concatenate([W_x] * 4, 0)),
                             h2h_weight_initializer=mx.init.Constant(np.concatenate([W_h] * 4, 0)),
                             h2h_bias_initializer=mx.init.Constant(b))
    wrapper = contrib.rnn.VariationalDropoutCell(lstm_cell, drop_states=dropout_h)
    return wrapper


def biLSTM(f_lstm, b_lstm, inputs, dropout_x=0.):
    """Feature extraction through BiLSTM

    Parameters
    ----------
    f_lstm : VariationalDropoutCell
        Forward cell
    b_lstm : VariationalDropoutCell
        Backward cell
    inputs : NDArray
        seq_len x batch_size
    dropout_x : float
        Variational dropout on inputs

    Returns
    -------
    outputs : NDArray
        Outputs of BiLSTM layers, seq_len x 2 hidden_dims x batch_size
    """
    for f, b in zip(f_lstm, b_lstm):
        inputs = nd.Dropout(inputs, dropout_x, axes=[0])  # important for variational dropout
        fo, _ = f.unroll(length=inputs.shape[0], inputs=inputs, layout='TNC', merge_outputs=True)
        bo, _ = b.unroll(length=inputs.shape[0], inputs=inputs.flip(axis=0), layout='TNC',
                         merge_outputs=True)
        f.reset()
        b.reset()
        inputs = nd.concat(fo, bo.flip(axis=0), dim=2)
    return inputs


def leaky_relu(x):
    """slope=0.1 leaky ReLu

    Parameters
    ----------
    x : NDArray
        Input

    Returns
    -------
    y : NDArray
        y = x > 0 ? x : 0.1 * x
    """
    return nd.LeakyReLU(x, slope=.1)


def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    """Do xWy

    Parameters
    ----------
    x : NDArray
        (input_size x seq_len) x batch_size
    W : NDArray
        (num_outputs x ny) x nx
    y : NDArray
        (input_size x seq_len) x batch_size
    input_size : int
        input dimension
    seq_len : int
        sequence length
    batch_size : int
        batch size
    num_outputs : int
        number of outputs
    bias_x : bool
        whether concat bias vector to input x
    bias_y : bool
        whether concat bias vector to input y

    Returns
    -------
    output : NDArray
        [seq_len_y x seq_len_x if output_size == 1 else seq_len_y x num_outputs x seq_len_x]
        x batch_size
    """
    if bias_x:
        x = nd.concat(x, nd.ones((1, seq_len, batch_size)), dim=0)
    if bias_y:
        y = nd.concat(y, nd.ones((1, seq_len, batch_size)), dim=0)

    ny = input_size + bias_y
    # W: (num_outputs x ny) x nx
    lin = nd.dot(W, x)
    if num_outputs > 1:
        lin = reshape_fortran(lin, (ny, num_outputs * seq_len, batch_size))
    y = y.transpose([2, 1, 0])  # May cause performance issues
    lin = lin.transpose([2, 1, 0])
    blin = nd.batch_dot(lin, y, transpose_b=True)
    blin = blin.transpose([2, 1, 0])
    if num_outputs > 1:
        blin = reshape_fortran(blin, (seq_len, num_outputs, seq_len, batch_size))
    return blin


def orthonormal_initializer(output_size, input_size, debug=False):
    """adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py

    Parameters
    ----------
    output_size : int
    input_size : int
    debug : bool
        Whether to skip this initializer
    Returns
    -------
    Q : np.ndarray
        The orthonormal weight matrix of input_size x output_size
    """
    print((output_size, input_size))
    if debug:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        return np.transpose(Q.astype(np.float32))
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for _ in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                np.abs(Q2 + Q2.sum(axis=0, keepdims=True)
                       + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print(('Orthogonal pretrainer loss: %.2e' % loss))
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def arc_argmax(parse_probs, length, tokens_to_keep, ensure_tree=True):
    """MST
    Adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py

    Parameters
    ----------
    parse_probs : NDArray
        seq_len x seq_len, the probability of arcs
    length : NDArray
        real sentence length
    tokens_to_keep : NDArray
        mask matrix
    ensure_tree :
        whether to ensure tree structure of output (apply MST)
    Returns
    -------
    parse_preds : np.ndarray
        prediction of arc parsing with size of (seq_len,)
    """
    if ensure_tree:
        I = np.eye(len(tokens_to_keep))
        # block loops and pad heads
        parse_probs = parse_probs * tokens_to_keep * (1 - I)
        parse_preds = np.argmax(parse_probs, axis=1)
        tokens = np.arange(1, length)
        roots = np.where(parse_preds[tokens] == 0)[0] + 1
        # ensure at least one root
        if len(roots) < 1:
            # The current root probabilities
            root_probs = parse_probs[tokens, 0]
            # The current head probabilities
            old_head_probs = parse_probs[tokens, parse_preds[tokens]]
            # Get new potential root probabilities
            new_root_probs = root_probs / old_head_probs
            # Select the most probable root
            new_root = tokens[np.argmax(new_root_probs)]
            # Make the change
            parse_preds[new_root] = 0
        # ensure at most one root
        elif len(roots) > 1:
            # The probabilities of the current heads
            root_probs = parse_probs[roots, 0]
            # Set the probability of depending on the root zero
            parse_probs[roots, 0] = 0
            # Get new potential heads and their probabilities
            new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) + 1
            new_head_probs = parse_probs[roots, new_heads] / root_probs
            # Select the most probable root
            new_root = roots[np.argmin(new_head_probs)]
            # Make the change
            parse_preds[roots] = new_heads
            parse_preds[new_root] = 0
        # remove cycles
        tarjan = Tarjan(parse_preds, tokens)
        for SCC in tarjan.SCCs:
            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)),
                            np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) + 1
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds
    else:
        # block and pad heads
        parse_probs = parse_probs * tokens_to_keep
        parse_preds = np.argmax(parse_probs, axis=1)
        return parse_preds


def rel_argmax(rel_probs, length, ensure_tree=True):
    """Fix the relation prediction by heuristic rules

    Parameters
    ----------
    rel_probs : NDArray
        seq_len x rel_size
    length :
        real sentence length
    ensure_tree :
        whether to apply rules
    Returns
    -------
    rel_preds : np.ndarray
        prediction of relations of size (seq_len,)
    """
    if ensure_tree:
        rel_probs[:, ParserVocabulary.PAD] = 0
        root = ParserVocabulary.ROOT
        tokens = np.arange(1, length)
        rel_preds = np.argmax(rel_probs, axis=1)
        roots = np.where(rel_preds[tokens] == root)[0] + 1
        if len(roots) < 1:
            rel_preds[1 + np.argmax(rel_probs[tokens, root])] = root
        elif len(roots) > 1:
            root_probs = rel_probs[roots, root]
            rel_probs[roots, root] = 0
            new_rel_preds = np.argmax(rel_probs[roots], axis=1)
            new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
            new_root = roots[np.argmin(new_rel_probs)]
            rel_preds[roots] = new_rel_preds
            rel_preds[new_root] = root
        return rel_preds
    else:
        rel_probs[:, ParserVocabulary.PAD] = 0
        rel_preds = np.argmax(rel_probs, axis=1)
        return rel_preds


def reshape_fortran(tensor, shape):
    """The missing Fortran reshape for mx.NDArray

    Parameters
    ----------
    tensor : NDArray
        source tensor
    shape : NDArray
        desired shape

    Returns
    -------
    output : NDArray
        reordered result
    """
    return tensor.T.reshape(tuple(reversed(shape))).T
