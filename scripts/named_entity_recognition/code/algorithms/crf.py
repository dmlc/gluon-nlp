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

#
# This is a reusable CRF module, Batch training has been implemented and
# the foreach operation has been used to improve the for loop unroll on the sequence in previous versions.
# Implemented with gluon, it can be encapsulated as Gluon's neural network layer and interface with other network layers.
#
# When constructing a CRF class, you need to specify the tag2idx and ctx,
# that is, whether the program runs on the CPU or GPU: the gpu is detected by default,
# and cpu is used if it is not detected.

# @author：kenjewu
# @date：2018/11/05


import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


CTX = try_gpu()
START_TAG = "<eos>"
STOP_TAG = "<bos>"


def log_sum_exp(vec):
    # max_score shape: （self.tagset_size, batch_size, 1)
    max_score = nd.max(vec, axis=-1, keepdims=True)
    # score shape: （self.tagset_size, batch_size, 1)
    score = nd.log(nd.sum(nd.exp(vec - max_score), axis=-1, keepdims=True)) + max_score

    # return NDArray shape: (self.tagset_size, batch_size, )
    return nd.squeeze(score, axis=-1)


class CRF(gluon.nn.Block):
    '''
    Custom CRF layer
    '''

    def __init__(self, tag2idx, ctx=CTX, **kwargs):
        '''Constructor

        Args:
            tag2idx (dict): a dictionary of sequence tags, such as: {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
            ctx (context of mxnet, optional): Defaults to CTX. 
        '''

        super(CRF, self).__init__(** kwargs)
        with self.name_scope():
            self.tag2idx = tag2idx
            self.ctx = ctx
            self.tagset_size = len(tag2idx)
            # Define the parameters of the transfer scoring matrix, where each value in
            # the matrix represents a score that transitions from state j to state i
            self.transitions = self.params.get(
                'transitions', shape=(self.tagset_size, self.tagset_size),
                init=mx.init.Xavier(magnitude=2.24))

    def _forward_alg(self, feats):
        '''Forward algorithm for CRF probability calculation

        Args:
            feats (NDArray): a representative representation of each symbol representing the entire sequence in a batch, 
                             shape is (seq_len, batch_size, tagset_size)

        Returns:
            alpha (NDArray): shape is (batch_size, )
        '''

        # Defining forward NDArray
        batch_size = feats.shape[1]
        # alphas shape: (batch_size, self.tagset_size)
        alphas = nd.full((batch_size, self.tagset_size), -10000., ctx=self.ctx)
        alphas[:, self.tag2idx[START_TAG]] = 0.

        def update_alphas(data, alphas):
            '''Calculate the batch update alpha for each time step

            Args:
                data (NDArray): NDArray shape: (seq_len, batch_size, self.tagset_size)
                alphas (NDArray): NDArray shape: (batch_size, self.tagset_size)
            '''

            # alphas_t shape: (self.tagset_size, batch_size, self.tagset_size)
            alphas_t = nd.broadcast_axis(nd.expand_dims(alphas, axis=0), axis=0, size=self.tagset_size)
            # emit_score shape: (self.tagset_size, batch_size, 1)
            emit_score = nd.transpose(nd.expand_dims(data, axis=0), axes=(2, 1, 0))
            # trans_score shape: (self.tagset_size, 1, self.tagset_size)
            trans_score = nd.expand_dims(self.transitions.data(), axis=1)

            # next_tag_var shape: (self.tagset_size, batch_size, self.tagset_size)
            next_tag_var = alphas_t + emit_score + trans_score

            # alphas shape: (self.tagset_size, batch_size)
            alphas = log_sum_exp(next_tag_var)
            # alphas shape: (batch_size, self.tagset_size)
            alphas = nd.transpose(alphas, axes=(1, 0))

            return data, alphas

        # Use the foreach operator to unroll
        _, alphas = nd.contrib.foreach(update_alphas, feats, alphas)

        # terminal_var shape:(batch_size, self.tagset_size)
        terminal_var = alphas + self.transitions.data()[self.tag2idx[STOP_TAG]]
        # alpha shape: (batch_size, )
        alpha = log_sum_exp(terminal_var)
        assert alpha.shape == (batch_size, )
        return alpha

    def _score_sentence(self, feats, tags):
        '''Calculate the score of the labeled sequence

        Args:
            feats (NDArray): a representative representation of each symbol representing the entire sequence in a batch, 
                             shape is (seq_len, batch_size, tagset_size)
            tags (NDArray): An index representing the label of each symbol in a batch, shape is (batch_size, seq_len)

        Returns:
            score (NDArray): value of label, shape is (batch_size, )
        '''

        batch_size = feats.shape[1]
        score = nd.zeros((batch_size,), ctx=self.ctx)

        # A matrix that retrieves the start tag of a batch of sentence symbol sequences. shape：(batch_size, 1)
        temp = nd.array([self.tag2idx[START_TAG]] * batch_size,
                        ctx=self.ctx).reshape((batch_size, 1))
        # concat, shape： (batch_size, seq_len+1)
        tags = nd.concat(temp, tags, dim=1)

        def update_score(data, states):
            '''计算评分

            Args:
                data (NDArray): NDArray shape:(seq_len, batch_size, self.tagset_size)
                states (list of NDArray): [idx, tags, score]

            Returns:
                score (NDArray): NDarray shape: (batch_size,)
            '''
            # feat shape: (batch_size, self.tagset_size)
            feat = data
            # tag shape:(batch_size, 1)
            idx, tags_iner, score = states
            i = int(idx.asscalar())
            score = score + nd.pick(self.transitions.data()[tags_iner[:, i + 1]],
                                    tags_iner[:, i], axis=1) + nd.pick(feat, tags_iner[:, i + 1], axis=1)
            idx += 1

            return feat, [idx, tags, score]

        states = [nd.array([0]), tags, score]
        _, states = nd.contrib.foreach(update_score, feats, states)
        score = states[2]
        score = score + self.transitions.data()[self.tag2idx[STOP_TAG], tags[:, int(tags.shape[1]-1)]]
        return score

    def _viterbi_decode(self, feats):
        '''
        CRF's prediction algorithm, Viterbi algorithm, which finds the best path based on features

        Args:
            feats (NDArray): a representative representation of each symbol representing the entire sequence in a batch, 
                             shape is (seq_len, batch_size, tagset_size)

        Returns:
            path_score (NDArray): value of score , shape is (batch_size, )
            best_path_matrix (NDArray): the matrix of best path, shape is (batch_size, seq_len)
        '''

        batch_size = feats.shape[1]

        # vvars shape：(batch_size, self.tagset_size)
        vvars = nd.full((batch_size, self.tagset_size), -10000., ctx=self.ctx)
        vvars[:, self.tag2idx[START_TAG]] = 0.0

        def update_decode(data, states):
            feat = data
            vvars_iner = states

            # vvars_t shape: (self.tagset_size, batch_size, self.tagset_size)
            vvars_t = nd.broadcast_axis(nd.expand_dims(vvars_iner, axis=0), axis=0, size=self.tagset_size)
            # trans shape: (self.tagset_size, 1, self.tagset_size)
            trans = nd.expand_dims(self.transitions.data(), axis=1)
            next_tag_var = vvars_t + trans

            # best_tag_id shape: (self.tagset_size, batch_size)
            best_tag_id = nd.argmax(next_tag_var, axis=-1)

            # bptrs_t, viterbivars_t  shape ：(batch_size, tagset_size)
            viterbivars_t = nd.transpose(nd.pick(next_tag_var, best_tag_id, axis=-1), axes=(1, 0))
            bptrs_t = nd.transpose(best_tag_id, axes=(1, 0))

            vvars_iner = viterbivars_t + feat

            return bptrs_t, vvars_iner

        # backpointers shape: (seq_len, batch_size, self.tagset_size)
        backpointers, vvars = nd.contrib.foreach(update_decode, feats, vvars)

        # transform to STOP_TAG
        # terminal_var shape: (batch_size, self.tagset_size)
        terminal_var = vvars + self.transitions.data()[self.tag2idx[STOP_TAG]]
        best_tag_id = nd.argmax(terminal_var, axis=1)
        # path_score shape:（batch_size, )
        path_score = nd.pick(terminal_var, best_tag_id, axis=1)

        # According to the reverse pointer backpointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in nd.reverse(backpointers, axis=0):
            # best_tag_id shape: (batch_size, )
            best_tag_id = nd.pick(bptrs_t, best_tag_id, axis=1)
            best_path.append(best_tag_id)

        # remove START_TAG
        # start shape: (batch_size, )
        start = best_path.pop()
        # Check if start is the start symbol
        for i in range(batch_size):
            assert start[i].asscalar() == self.tag2idx[START_TAG]
        best_path.reverse()

        # Build the matrix of the best path, shape: (batch_size, seq_len)
        best_path_matrix = nd.stack(*best_path, axis=1)
        return path_score, best_path_matrix

    def neg_log_likelihood(self, feats, tags):
        '''Calculate the log likelihood of the tag sequence in the CRF, ie crf_loss

        Args:
            feats (NDArray): a representative representation of each symbol representing the entire sequence in a batch, 
                             shape is (seq_len, batch_size, tagset_size)
            tags (NDArray): An index representing the label of each symbol in a batch, shape is (batch_size, seq_len)

        Returns:
            [NDArray]: loss, shape: (batch_size, )
        '''

        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        # shape： (batch_size, )
        return forward_score - gold_score

    def forward(self, feats):
        '''Use CRF to predict results based on input characteristics, which can be batched

        Args:
            feats (NDArray): a representative representation of each symbol representing the entire sequence in a batch, 
                             shape is (seq_len, batch_size, tagset_size)

        Returns:
            score (NDArray): value of score , shape is (batch_size, )
            tag_seq (NDArray): the matrix of best path, shape is (batch_size, seq_len)
        '''

        score, tag_seq = self._viterbi_decode(feats)

        return score, tag_seq
