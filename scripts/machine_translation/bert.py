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
"""Machine translation models and translators."""


__all__ = ['BERTModel']

import warnings
import numpy as np
from mxnet.gluon import Block
from mxnet.gluon import nn
import mxnet as mx
from gluonnlp.model import BeamSearchScorer, BeamSearchSampler


class BERTModel(Block):
    """Model for Neural Machine Translation.

    Parameters
    ----------
    src_vocab : Vocab
        Source vocabulary.
    tgt_vocab : Vocab
        Target vocabulary.
    encoder : Seq2SeqEncoder
        Encoder that encodes the input sentence.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the source and target embeddings
        if src_embed and tgt_embed are None.
    embed_dropout : float, default 0.0
        Dropout rate of the embedding weights. It is used to generate the source and target
        embeddings if src_embed and tgt_embed are None.
    embed_initializer : Initializer, default mx.init.Uniform(0.1)
        Initializer of the embedding weights. It is used to generate ghe source and target
        embeddings if src_embed and tgt_embed are None.
    src_embed : Block or None, default None
        The source embedding. If set to None, src_embed will be constructed using embed_size and
        embed_dropout.
    tgt_embed : Block or None, default None
        The target embedding. If set to None and the tgt_embed will be constructed using
        embed_size and embed_dropout. Also if `share_embed` is turned on, we will set tgt_embed
        to be the same as src_embed.
    share_embed : bool, default False
        Whether to share the src/tgt embeddings or not.
    tgt_proj : Block or None, default None
        Layer that projects the decoder outputs to the target vocabulary.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
        See document of `Block`.
    """
    def __init__(self, src_vocab, tgt_vocab, encoder,
                 embed_size=None, embed_dropout=0.0, embed_initializer=mx.init.Uniform(0.1),
                 src_embed=None, tgt_embed=None, share_embed=False, tie_weights=False,
                 tgt_proj=None, prefix=None, params=None):
        super(BERTModel, self).__init__(prefix=prefix, params=params)
        self.tgt_vocab = tgt_vocab
        self.src_vocab = src_vocab
        self.encoder = encoder
        self._shared_embed = share_embed
        if embed_dropout is None:
            embed_dropout = 0.0
        # Construct src embedding
        if share_embed and tgt_embed is not None:
            warnings.warn('"share_embed" is turned on and \"tgt_embed\" is not None. '
                          'In this case, the provided "tgt_embed" will be overwritten by the '
                          '"src_embed". Is this intended?')
        if src_embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "src_embed" is not ' \
                                           'given.'
            with self.name_scope():
                self.src_embed = nn.HybridSequential(prefix='src_embed_')
                with self.src_embed.name_scope():
                    self.src_embed.add(nn.Embedding(input_dim=len(src_vocab), output_dim=embed_size,
                                                    weight_initializer=embed_initializer))
                    self.src_embed.add(nn.Dropout(rate=embed_dropout))
        else:
            self.src_embed = src_embed
        # Construct tgt embedding
        if share_embed:
            self.tgt_embed = self.src_embed
        else:
            if tgt_embed is not None:
                self.tgt_embed = tgt_embed
            else:
                assert embed_size is not None,\
                    '"embed_size" cannot be None if "tgt_embed" is ' \
                    'not given and "shared_embed" is not turned on.'
                with self.name_scope():
                    self.tgt_embed = nn.HybridSequential(prefix='tgt_embed_')
                    with self.tgt_embed.name_scope():
                        self.tgt_embed.add(
                            nn.Embedding(input_dim=len(tgt_vocab), output_dim=embed_size,
                                         weight_initializer=embed_initializer))
                        self.tgt_embed.add(nn.Dropout(rate=embed_dropout))
            # Construct tgt proj
        if tie_weights:
            self.tgt_proj = nn.Dense(units=len(tgt_vocab), flatten=False,
                                     params=self.tgt_embed.params, prefix='tgt_proj_')
        else:
            if tgt_proj is None:
                with self.name_scope():
                    self.tgt_proj = nn.Dense(units=len(tgt_vocab), flatten=False,
                                             prefix='tgt_proj_')
            else:
                self.tgt_proj = tgt_proj

    def encode(self, inputs, states=None, valid_length=None):
        """Encode the input sequence.

        Parameters
        ----------
        inputs : NDArray
        states : list of NDArrays or None, default None
        valid_length : NDArray or None, default None

        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        return self.encoder(self.src_embed(inputs), states, valid_length)

    def __call__(self, src_seq, src_valid_length=None):  #pylint: disable=arguments-differ
        """Generate the prediction given the src_seq.

        This is used in training an NMT model.

        Parameters
        ----------
        src_seq : NDArray
        src_valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, tgt_length, tgt_word_num)
        additional_outputs : list of list
            Additional outputs of encoder and decoder, e.g, the attention weights
        """
        return super(BERTModel, self).__call__(src_seq, src_valid_length)

    def forward(self, src_seq, src_valid_length=None):  #pylint: disable=arguments-differ
        """Generate the prediction given the src_seq.

        This is used in training an NMT model.

        Parameters
        ----------
        src_seq : NDArray
        src_valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, src_length, C_out)
        additional_outputs : list of list
            Additional outputs of encoder and decoder, e.g, the attention weights
        """
        additional_outputs = []
        outputs, encoder_additional_outputs = self.encode(src_seq,
                                                          valid_length=src_valid_length)
        additional_outputs.append(encoder_additional_outputs)
        return outputs, additional_outputs
