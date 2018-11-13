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
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon import nn
import mxnet as mx
from . import TransformerEncoder


class BERTModel(Block):
    """Model for Bidirectional Encoder Representations from Transformers.

    Parameters
    ----------
    encoder : TransformerEncoder
        Bidirectional encoder that encodes the input sentence.
    vocab_size : int or None, default None
        The size of the vocabulary.
    token_type_vocab_size : int or None, default None
        The vocabulary size of token types.
    units: int or None, default None
        Number of units for the final pooler layer.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the word and token type
        embeddings if word_embed and token_type_embed are None.
    embed_dropout : float, default 0.0
        Dropout rate of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    embed_initializer : Initializer, default mx.init.Normal(0.02)
        Initializer of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    word_embed : Block or None, default None
        The word embedding. If set to None, word_embed will be constructed using embed_size and
        embed_dropout.
    token_type_embed : Block or None, default None
        The token type embedding. If set to None and the token_type_embed will be constructed using
        embed_size and embed_dropout.
    pooler : Block or None, default None
        Layer that projects the encoded tensor for the first token in the sequence.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """
    def __init__(self, encoder, vocab_size=None, token_type_vocab_size=None, units=None,
                 embed_size=None, embed_dropout=0.0, embed_initializer=mx.init.Normal(0.02),
                 word_embed=None, token_type_embed=None,
                 pooler=None, prefix=None, params=None):
        super(BERTModel, self).__init__(prefix=prefix, params=params)
        self.encoder = encoder
        # Construct word embedding
        self.word_embed = self._get_embed(word_embed, vocab_size, embed_size,
                                          embed_initializer, embed_dropout, 'word_embed_')
        # Construct token type embedding
        self.token_type_embed = self._get_embed(token_type_embed, token_type_vocab_size,
                                                embed_size, embed_initializer, embed_dropout,
                                                'token_type_embed_')
        # Construct pooler
        self.pooler = self._get_pooler(pooler, units, 'pooler_')

    def _get_embed(self, embed, vocab_size, embed_size, initializer, dropout, prefix):
        """ Construct an embedding block. """
        if embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "word_embed" or ' \
                                           'token_type_embed is not given.'
            with self.name_scope():
                embed = nn.HybridSequential(prefix=prefix)
                with embed.name_scope():
                    # TODO(haibin) initialize with truncated normal if trained from scratch
                    embed.add(nn.Embedding(input_dim=vocab_size, output_dim=embed_size,
                                           weight_initializer=initializer))
                    if dropout:
                        embed.add(nn.Dropout(rate=dropout))
        assert isinstance(embed, Block)
        return embed

    def _get_pooler(self, pooler, units, prefix):
        """ Construct pooler.

        The pooler slices and projects the hidden output of first token
        in the sequence for segment level classification.

        """
        if pooler is None:
            assert units is not None, '"units" cannot be None if "pooler" is not given.'
            with self.name_scope():
                pooler = nn.Dense(units=units, flatten=False, activation='tanh',
                                  prefix=prefix)
        assert isinstance(pooler, Block)
        return pooler

    def forward(self, inputs, token_types, valid_length=None): #pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.

        Parameters
        ----------
        inputs : NDArray
        token_types : NDArray
        valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, sequence_length, C_out)
        additional_outputs : list
            Additional outputs of the encoder, e.g, the attention weights
        """
        return self.get_sequence_output(inputs, token_types, valid_length)

    def get_sequence_output(self, inputs, token_types, valid_length=None):
        #pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.

        Parameters
        ----------
        inputs : NDArray
        token_types : NDArray
        valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, sequence_length, C_out)
        additional_outputs : list
            Additional outputs of the encoder, e.g, the attention weights
        """
        # embedding
        word_embedding = self.word_embed(inputs)
        type_embedding = self.token_type_embed(token_types)
        embedding = word_embedding + type_embedding
        # encoding
        outputs, additional_outputs = self.encoder(embedding, None, valid_length)
        return outputs, additional_outputs

    def get_pooled_output(self, inputs, token_types, valid_length=None): #pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.

        Parameters
        ----------
        inputs : NDArray
        token_types : NDArray
        valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, C_out)
        additional_outputs : list
            Additional outputs of the encoder, e.g, the attention weights
        """
        outputs, additional_outputs = self.get_sequence_output(inputs, token_types, valid_length)
        # pooling
        outputs = outputs.slice(begin=(None, 0, None), end=(None, 1, None)).squeeze()
        outputs = self.pooler(outputs)
        return outputs, additional_outputs

def get_transformer_encoder(num_layers=12,
                            num_heads=12, scaled=True,
                            units=768, hidden_size=3072, dropout=0.0, 
                            positional_weight='learned', use_residual=True,
                            max_src_length=50, max_tgt_length=50,
                            ffn_activation='gelu', attention_use_bias=True,
                            attention_proj_use_bias=True,
                            weight_initializer=None, bias_initializer='zeros',
                            layer_norm_eps=1e-12,
                            prefix='transformer_', params=None):
    """Build transformer encoder for BERT

    Parameters
    ----------
    num_layers : int
    num_heads : int
    scaled : bool
    units : int
    hidden_size : int
    dropout : float
    use_residual : bool
    max_src_length : int
    max_tgt_length : int
    ffn_activation : str, default is 'relu'
    attention_use_bias : bool
        Apply bias term when projecting key, value, query in the attention cell. Default is True.
    attention_proj_use_bias : bool
        Apply bias term to the linear projection of the output of attention cell. Default is False.
    weight_initializer : mx.init.Initializer or None
    bias_initializer : mx.init.Initializer or None
    prefix : str, default 'transformer_'
        Prefix for name of `Block`s.
    params : Parameter or None
        Container for weight sharing between layers.
        Created if `None`.

    Returns
    -------
    encoder : TransformerEncoder
    """
    encoder = TransformerEncoder(num_layers=num_layers,
                                 num_heads=num_heads,
                                 max_length=max_src_length,
                                 units=units,
                                 hidden_size=hidden_size,
                                 dropout=dropout,
                                 scaled=scaled,
                                 positional_weight=positional_weight,
                                 use_residual=use_residual,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 ffn_activation=ffn_activation,
                                 attention_use_bias=attention_use_bias,
                                 attention_proj_use_bias=attention_proj_use_bias,
                                 layer_norm_eps=layer_norm_eps,
                                 apply_layernorm_before_dropout=True,
                                 prefix=prefix + 'enc_', params=params)
    return encoder
