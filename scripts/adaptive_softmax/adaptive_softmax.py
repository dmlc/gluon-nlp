"""
We implement the adaptive softmax proposed in the following work:
@article{grave2016efficient,
         title={Efficient softmax approximation for GPUs},
         author={Grave, Edouard and Joulin, Armand and Ciss{\'e},
                 Moustapha and Grangier, David and J{\'e}gou, Herv{\'e}},
         journal={arXiv preprint arXiv:1609.04309},
         year={2016}
}
"""

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

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

class Adaptivesoftmax(gluon.Block):
    """
    Implementation of the adaptive softmax proposed in the following work:

    Parameters:
    ----------
    input_size: int
       the input size for adaptive softmax function, which is defined as the output size of
       hidden layers (num_hidden).
    cutoff: list or np.array
       build clusters for adaptive softmax.
    """
    def __init__(self, input_size, cutoff, reduce_factor=4):
        super(Adaptivesoftmax, self).__init__()

        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1

        self.head = nn.Dense(units=self.output_size, in_units=input_size, flatten=False)
        self.tail = nn.Sequential()

        for i in range(len(cutoff) - 1):
            if reduce_factor == 1:
                seq = nn.Dense(units=(cutoff[i + 1] - cutoff[i]), in_units=input_size,
                               flatten=False)

            else:
                seq = nn.Sequential()
                seq.add(nn.Dense(units=(input_size // reduce_factor ** i),
                                 in_units=input_size, flatten=False))
                seq.add(nn.Dense(units=(cutoff[i + 1] - cutoff[i]),
                                 in_units=(input_size // reduce_factor ** i), flatten=False))

            self.tail.add(seq)

    def set_target(self, target):
        """
        Generate id array to assgin the sample to the specific clulster it belongs to.
        As this part is requires no gradient update, the target is transformed into
        numpy for calculation.

        Parameters:
        ----------
        target: NDArray
           the label for training and its shape is `(batch_size * bptt, 1)`.

        Returns
        ----------
        self.id: numpy
           the id array can be used to assign the sample to the specific cluster it belongs to
        """
        self.id = []
        target = target.asnumpy()

        for i in range(len(self.cutoff) - 1):
            mask_1 = (target >= self.cutoff[i])
            mask_2 = (target <= self.cutoff[i + 1])
            mask = mask_1 * mask_2

            mask = mask.reshape((mask.shape[1], mask.shape[0]))
            if True in mask:
                self.id.append(np.where(mask[0])[0])

            else:
                self.id.append(None)

    def remap_target(self, target):
        """
        Map the target according to the different clusters they belong to.
        new_target[0] refers to the 'head'
        new_target[1], .... refer to the 'tail'
        As this part requires no gradient update, the target is transformed into numpy
        for calculation.

        Parameters:
        ----------
        target: NDArray
           the label for training and its shape is `(batch_size * bptt, 1)`.

        Returns
        ----------
        new_target: numpy
            The new_target array can be used to assign the respective target to the sample
        """
        target = target.asnumpy()
        new_target = []
        new_target.append(np.copy(target))

        for i in range(len(self.cutoff) - 1):
            mask_1 = (target >= self.cutoff[i])
            mask_2 = (target <= self.cutoff[i + 1])
            mask = mask_1 * mask_2
            new_target[0][mask] = self.cutoff[0] + i

            if True in mask:
                new_target.append(target[mask] - self.cutoff[i])

            else:
                new_target.append(None)

        return new_target


    def forward(self, inputs, target):
        """
        If adaptive softmax is true, this function will be called.

        Parameters:
        ----------
        input: NDArray
           the output from the hidden layers and its shape is `(batch_size * bptt, num_hidden)`.
        target: NDArray
           the label for training and its shape is `(batch_size * bptt, 1)`.

        Returns
        ----------
        nnloss: NDArray
            The output is the loss.
        """
        output_head = self.head(inputs)
        nnloss = 0
        self.target = target
        context = inputs.context

        if self.target is not None:
            self.set_target(self.target)
            self.target = self.remap_target(self.target)

        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        nnloss = nnloss + \
                 mx.nd.sum(loss(output_head, mx.nd.array(self.target[0]).as_in_context(context)))

        for i in range(len(self.id)):
            if self.id[i] is not None:
                id_select = np.array(self.id[i])
                output_tail = self.tail[i](inputs[id_select])
                nnloss = nnloss + \
                         mx.nd.sum(loss(output_tail, \
                         mx.nd.array(self.target[i+1]).as_in_context(context)))

        nnloss = nnloss / (len(target))

        return nnloss

    def log_prob(self, inputs):
        """
        If adaptive softmax is false, this function will be called.

        Parameters:
        ----------
        input: NDArray
           the output from the hidden layers and its shape is `(batch_size * bptt, num_hidden)`.

        Returns
        ----------
        prob: NDArray
            The array contains the probability for the final output pontential word.
            Its shape is `(batch_size * bptt, vocab_size + 1)`.
        """
        head_out = self.head(inputs)
        target_size = head_out.shape[0]
        prob = mx.nd.zeros((target_size, self.cutoff[-1]))

        lsm_head = mx.nd.log_softmax(head_out, axis=1)
        prob[:, : self.cutoff[0]] = lsm_head[:, : self.cutoff[0]]

        for i in range(len(self.tail)):
            split = lsm_head[:, self.cutoff[0] + i]
            split = split.expand_dims(1)
            tail_out = self.tail[i](inputs)
            if i == 10:
                print(tail_out[0])
                print('tail loss:', mx.nd.log_softmax(tail_out, axis=1)[-1][3395])
            lsm_tail = mx.nd.log_softmax(tail_out, axis=1) + \
                       split.broadcast_to((tail_out.shape[0], tail_out.shape[1]))
            prob[:, self.cutoff[i] : self.cutoff[i + 1]] = lsm_tail

        return prob
