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

from mxnet import gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
from adaptive_softmax import Adaptivesoftmax

class LanguageModel(gluon.Block):
    """LanguageModel for adaptive softmax and regular full softmax.

    Parameters:
    ----------
    vocab_size: int
       the size of the vocabulary.
    num_embed: int
       the size of the embedding layer.
    num_hidden: int
       the size of the hidden layer.
    num_layers: int
       the number of hidden layers.
    dropout: float
       the chance of one connection to be ignored.
    adaptive_softmax: bool
       If it is "True", the model uses adaptive softmax and the function "forward" will be called.
       If it is "False", the model uses regular full softmax and the the function "log_prob" will
       be called.
    ctx:
       Calculation is based on mx.gpu(0) or mx.cpu(0)
    cutoff: list or np.array
       Build clusters for adaptive softmax.
    """
    def __init__(self, vocab_size, num_embed, num_hidden, num_layers, cutoff,
                 dropout=0.0, adaptive_softmax=True, ctx=mx.gpu(0), **kwargs):
        super(LanguageModel, self).__init__(**kwargs)
        self.ctx = ctx

        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer=mx.init.Uniform(0.1))

            self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                input_size=num_embed)

        if adaptive_softmax:
            self.linear = Adaptivesoftmax(num_hidden, [*cutoff, vocab_size + 1])
        else:
            self.linear = nn.Dense(units=vocab_size, in_units=num_hidden, flatten=False)

        self.adaptive_softmax = adaptive_softmax

        self.num_layers = num_layers
        self.num_hidden = num_hidden

    def forward(self, inputs, hidden, target=None):
        """Conduct 'forward' for NN

        Parameters:
        ----------
        inputs: NDArray
           The inputs from the samples.
        hidden: int
           The hidden layer for rnn.
        target: NDArray
           The target label.

        Returns
        ----------
        nnloss: NDArray
            The output is the loss.
        hidden: NDArray
            The hidden state for rnn.
        """
        embed = self.encoder(inputs)
        embed = self.drop(embed)

        output, hidden = self.rnn(embed, hidden)
        output = self.drop(output)

        if self.adaptive_softmax:
            self.linear.set_target(target)
            nnloss = self.linear(output.reshape((output.shape[0] * output.shape[1],
                                                 output.shape[2])), target)

        if not self.adaptive_softmax:
            output = self.linear(output.reshape((output.shape[0] * output.shape[1],
                                                 output.shape[2])))
            loss = gluon.loss.SoftmaxCrossEntropyLoss()
            nnloss = mx.nd.sum(loss(output, target))
            nnloss = nnloss / (len(target))

        return nnloss, hidden

    def log_prob(self, inputs, hidden):
        """Calculate the log probability of the'prediction

        Parameters:
        ----------
        inputs: NDArray
           The inputs from the samples.
        hidden: int
           The hidden layer for rnn.

        Returns
        ----------
        prob: NDArray
            It contains the probability for each word.
        hidden: NDArray
            The hidden state for rnn.
        """
        embed = self.encoder(inputs)
        output, hidden = self.rnn(embed, hidden)
        prob = self.linear.log_prob(output.reshape((output.shape[0] * output.shape[1],
                                                    output.shape[2])))

        return prob, hidden

    def begin_state(self, *args, **kwargs):
        """The initialization of the state of rnn

        Parameters:
        ----------
            The parameters here are the same as general case.
        """
        return self.rnn.begin_state(*args, **kwargs)
