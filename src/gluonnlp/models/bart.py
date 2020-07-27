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
"""
BART Model

@article{lewis2019bart,
    title = {BART: Denoising Sequence-to-Sequence Pre-training for Natural
Language Generation, Translation, and Comprehension},
    author = {Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad and
              Abdelrahman Mohamed and Omer Levy and Veselin Stoyanov
              and Luke Zettlemoyer },
    journal={arXiv preprint arXiv:1910.13461},
    year = {2019},
}

"""

import os
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn
from ..registry import BACKBONE_REGISTRY
from .transformer import TransformerEncoder, TransformerDecoder
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..initializer import TruncNorm
from ..attention_cell import MultiHeadAttentionCell, gen_self_attn_mask
from ..layers import get_activation, PositionalEmbedding, PositionwiseFFN, InitializerType
from ..op import select_vectors_by_position
from ..data.tokenizers import HuggingFaceWordPieceTokenizer

PRETRAINED_URL = {
    'fairseq_bart_base': {
        'cfg': bart_base(),
        'merges': 'fairseq_bart_base/',
        'vocab': 'fairseq_bart_base/',
        'params': 'fairseq_bart_base/',
        'mlm_params': 'fairseq_bart_base/',
        'lowercase': False,
    },
    'fairseq_bart_large': {
        'cfg': bart_large(),
        'merges': 'fairseq_bart_large/',
        'vocab': 'fairseq_bart_large/',
        'params': 'fairseq_bart_large/',
        'mlm_params': 'fairseq_bart_large/',
        'lowercase': False,
    }
}


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'bart.txt'))
bart_cfg_reg = Registry('bart_cfg')


@bart_cfg_reg.register()
def bart_base():
    cfg = CN()
    # Config for the bart base model
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 50265
    cfg.MODEL.max_length = 1024
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.shared_embed = True
    cfg.MODEL.tie_weights = True
    cfg.MODEL.attention_dropout_prob = 0.0
    cfg.MODEL.activation_dropout = 0.0
    cfg.MODEL.hidden_dropout_prob = 0.1
    cfg.MODEL.layer_norm_eps = 1E-5
    cfg.MODEL.pooler_activation = 'tanh'
    cfg.MODEL.dtype = 'float32'

    # Parameters for the encoder
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.num_layers = 6
    cfg.MODEL.ENCODER.units = 768
    cfg.MODEL.ENCODER.num_heads = 12
    cfg.MODEL.ENCODER.hidden_size = 3072
    cfg.MODEL.ENCODER.recurrent = False
    cfg.MODEL.ENCODER.activation = 'gelu'
    cfg.MODEL.ENCODER.pre_norm = False

    # Parameters for the decoder
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.num_layers = 6
    cfg.MODEL.DECODER.units = 768
    cfg.MODEL.DECODER.num_heads = 12
    cfg.MODEL.DECODER.hidden_size = 3072
    cfg.MODEL.DECODER.recurrent = False
    cfg.MODEL.DECODER.activation = 'gelu'
    cfg.MODEL.DECODER.pre_norm = False

    # Parameters for the initializer
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@bart_cfg_reg.register()
def bart_large():
    cfg = bart_base()
    cfg.defrost()
    cfg.MODEL.ENCODER.units = 1024
    cfg.MODEL.ENCODER.hidden_size = 4096
    cfg.MODEL.ENCODER.num_heads = 16
    cfg.MODEL.ENCODER.num_layers = 12
    cfg.MODEL.DECODER.units = 1024
    cfg.MODEL.DECODER.hidden_size = 4096
    cfg.MODEL.DECODER.num_heads = 16
    cfg.MODEL.DECODER.num_layers = 12
    cfg.freeze()
    return cfg

@use_np
class BART(HybridBlock):
    def __init__(self, src_vocab_size: int,
                 tgt_vocab_size: int,
                 max_src_length: Optional[int] = None,
                 max_tgt_length: Optional[int] = None,
                 scale_embed: bool = True,
                 pos_embed_type="sinusoidal",
                 shared_embed: bool = True,
                 tie_weights: bool = True,
                 activation_dropout: float = 0.0,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 layer_norm_eps: float = 1E-5,
                 data_norm: bool = False,
                 enc_units: int = 512,
                 enc_hidden_size: int = 2048,
                 enc_num_heads: int = 8,
                 enc_num_layers: int = 6,
                 enc_recurrent: bool = False,
                 enc_activation='relu',
                 enc_pre_norm: bool = False,
                 dec_units: int = 512,
                 dec_hidden_size: int = 2048,
                 dec_num_heads: int = 8,
                 dec_num_layers: int = 6,
                 dec_recurrent: bool = False,
                 dec_activation='relu',
                 dec_pre_norm: bool = False,
                 embed_initializer=TruncNorm(stdev=0.02),
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 dtype='float32'):
        """

        Parameters
        ----------
        src_vocab_size
            The vocabulary size of the source language
        tgt_vocab_size
            The vocabulary size of the target language
        max_src_length
            The maximal length of the source sequence.
            If it's negative, we will use treat it as not set.
        max_tgt_length
            The maximal length of the target sequence.
            If it's negative, we will use treat it as not set.
        scale_embed
            Whether to multiply the src and dst embeddings by sqrt(units)
        pos_embed_type
            Type of the positional embedding
        shared_embed
            Whether to share the embedding of the src and tgt language
        tie_weights
            Whether to tie the weights of input + output.
        activation_dropout
            The ratio of the activation dropout in FFN
        dropout
            The default dropout ratio
        attention_dropout
            The ratio of the attention dropout
        layer_norm_eps
            The epsilon of the layer normalization
        data_norm
            Whether to add layer normalization layer after the input.
        enc_units
            Units of the encoder
        enc_hidden_size
            Hidden size of the encoder
        enc_num_heads
            Number of heads of the encoder
        enc_num_layers
            Number of layers of the encoder
        enc_recurrent
            Whether to use recurrent encoder (share weights)
        enc_activation
            Activation of the encoder layer
        enc_pre_norm
            Whether to add layer_norm before self-attention in the encoder
        dec_units
            Units of the decoder
        dec_hidden_size
            Hidden size of the decoder
        dec_num_heads
            Number of heads of the decoder
        dec_num_layers
            Number of layers of the decoder
        dec_recurrent
            Whether to use recurrent decoder (share weights)
        dec_activation
            Activation of the decoder layer
        dec_pre_norm
            Whether to add layer_norm before self-attention in the decoder
        embed_initializer
            Initializer of the embedding layer
        weight_initializer
            Initializer of the weight
        bias_initializer
            Initializer of the bias
        dtype
            Data type of the weights
        """
        super().__init__()
        assert src_vocab_size > 0 and tgt_vocab_size > 0,\
            'Cannot set "src_vocab_size" and "tgt_vocab_size" to negative numbers. ' \
            'Are you creating ' \
            'the model with the config from TransformerModel.get_cfg()? If that is ' \
            'the case, you will need to set the cfg.MODEL.src_vocab_size and ' \
            'cfg.MODEL.tgt_vocab_size manually before passing to ' \
            'TransformerModel.from_cfg().'
        self._dtype = dtype
        self._src_vocab_size = src_vocab_size
        self._tgt_vocab_size = tgt_vocab_size
        self.pos_embed_type = pos_embed_type
        self.scaled_embed = scale_embed
        self.enc_units = enc_units
        self.dec_units = dec_units
        if max_src_length is not None and max_src_length < 0:
            max_src_length = None
        if max_tgt_length is not None and max_tgt_length < 0:
            max_tgt_length = None
        if enc_units != dec_units:
            assert shared_embed is False, 'Cannot share embedding when the enc_units and dec_units ' \
                                          'are different! enc_units={},' \
                                          ' dec_units={}'.format(enc_units, dec_units)
        self.src_embed_layer = nn.Embedding(input_dim=src_vocab_size,
                                            output_dim=enc_units,
                                            weight_initializer=embed_initializer,
                                            dtype=self._dtype)
        self.tgt_embed_layer = nn.Embedding(input_dim=tgt_vocab_size,
                                            output_dim=dec_units,
                                            weight_initializer=embed_initializer,
                                            dtype=self._dtype)
        if shared_embed:
            self.tgt_embed_layer.weight = self.src_embed_layer.weight
        if pos_embed_type is not None:
            self.src_pos_embed_layer = PositionalEmbedding(units=enc_units,
                                                           max_length=max_src_length,
                                                           dtype=self._dtype,
                                                           method=pos_embed_type)
            self.tgt_pos_embed_layer = PositionalEmbedding(units=dec_units,
                                                           max_length=max_tgt_length,
                                                           dtype=self._dtype,
                                                           method=pos_embed_type)
        self.encoder = TransformerEncoder(num_layers=enc_num_layers,
                                          recurrent=enc_recurrent,
                                          units=enc_units,
                                          hidden_size=enc_hidden_size,
                                          num_heads=enc_num_heads,
                                          activation_dropout=activation_dropout,
                                          dropout=dropout,
                                          attention_dropout=attention_dropout,
                                          layer_norm_eps=layer_norm_eps,
                                          weight_initializer=weight_initializer,
                                          bias_initializer=bias_initializer,
                                          activation=enc_activation,
                                          data_norm=data_norm,
                                          pre_norm=enc_pre_norm,
                                          dtype=self._dtype)
        self.decoder = TransformerDecoder(num_layers=dec_num_layers,
                                          recurrent=dec_recurrent,
                                          units=dec_units,
                                          mem_units=enc_units,
                                          hidden_size=dec_hidden_size,
                                          num_heads=dec_num_heads,
                                          activation_dropout=activation_dropout,
                                          dropout=dropout,
                                          attention_dropout=attention_dropout,
                                          layer_norm_eps=layer_norm_eps,
                                          weight_initializer=weight_initializer,
                                          bias_initializer=bias_initializer,
                                          activation=dec_activation,
                                          data_norm=data_norm,
                                          pre_norm=dec_pre_norm,
                                          dtype=self._dtype)
        if tie_weights:
            self.tgt_final_layer =\
                nn.Dense(tgt_vocab_size, flatten=False,
                         bias_initializer=bias_initializer,
                         use_bias=False,
                         dtype=self._dtype)
            self.tgt_final_layer.weight = self.tgt_embed_layer.weight
        else:
            self.tgt_final_layer = \
                nn.Dense(tgt_vocab_size,
                         flatten=False,
                         weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer,
                         use_bias=False,
                         dtype=self._dtype)
        self.encoder.hybridize()
        self.decoder.hybridize()

    @property
    def src_vocab_size(self):
        return self._src_vocab_size

    @property
    def tgt_vocab_size(self):
        return self._tgt_vocab_size

    # TODO(sxjscience) We can actually try to hybridize this function via the
    #  newly-introduced deferred compute.
    def encode(self, F, src_data, src_valid_length):
        """Encode the source data to memory

        Parameters
        ----------
        F
        src_data :
            Shape (batch_size, src_length)
        src_valid_length :
            Shape (batch_size,)

        Returns
        -------
        enc_out :
            Shape (batch_size, src_length, C_out)
        """
        src_data = self.src_embed_layer(src_data)
        if self.scaled_embed:
            src_data = src_data * np.sqrt(self.enc_units)
        if self.pos_embed_type is not None:
            src_data = src_data + self.src_pos_embed_layer(F.npx.arange_like(src_data, axis=1))
        enc_out = self.encoder(src_data, src_valid_length)
        return enc_out

    def decode_seq(self, F, tgt_data, tgt_valid_length, mem_data, mem_valid_length):
        """Decode a sequence of inputs

        Parameters
        ----------
        F
        tgt_data :
            Shape (batch_size, tgt_length)
        tgt_valid_length :
            Shape (batch_size,)
        mem_data :
            Shape (batch_size, src_length, C_out)
        mem_valid_length :
            Shape (batch_size,)

        Returns
        -------
        dec_out :
            Shape (batch_size, tgt_length, tgt_vocab_size)
        """
        tgt_data = self.tgt_embed_layer(tgt_data)
        if self.scaled_embed:
            tgt_data = tgt_data * np.sqrt(self.dec_units)
        if self.pos_embed_type is not None:
            tgt_data = tgt_data + self.tgt_pos_embed_layer(
                F.npx.arange_like(tgt_data, axis=1))
        dec_out = self.decoder(tgt_data, tgt_valid_length, mem_data, mem_valid_length)
        dec_out = self.tgt_final_layer(dec_out)
        return dec_out

    def hybrid_forward(self, F, src_data, src_valid_length, tgt_data, tgt_valid_length):
        """

        Parameters
        ----------
        F
        src_data :
            Shape (batch_size, src_length)
        src_valid_length :
            Shape (batch_size,)
        tgt_data :
            Shape (batch_size, tgt_length)
        tgt_valid_length :
            Shape (batch_size,)

        Returns
        -------
        out :
            Shape (batch_size, tgt_length, tgt_vocab_size)
        """
        enc_out = self.encode(F, src_data, src_valid_length)
        dec_out = self.decode_seq(F, tgt_data, tgt_valid_length, enc_out, src_valid_length)
        return dec_out

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            return bart_base()
        else:
            return bart_cfg_reg.create(key)

    @classmethod
    def from_cfg(cls, cfg):
        cfg = cls.get_cfg().clone_merge(cfg)
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        return cls(src_vocab_size=cfg.MODEL.vocab_size,
                   tgt_vocab_size=cfg.MODEL.vocab_size,
                   max_src_length=cfg.MODEL.max_length,
                   max_tgt_length=cfg.MODEL.max_length,
                   scale_embed=cfg.MODEL.scale_embed,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   shared_embed=cfg.MODEL.shared_embed,
                   tie_weights=cfg.MODEL.tie_weights,
                   attention_dropout=cfg.MODEL.attention_dropout,
                   activation_dropout=cfg.MODEL.activation_dropout,
                   dropout=cfg.MODEL.dropout,
                   enc_num_layers=cfg.MODEL.ENCODER.num_layers,
                   enc_units=cfg.MODEL.ENCODER.units,
                   enc_num_heads=cfg.MODEL.ENCODER.num_heads,
                   enc_hidden_size=cfg.MODEL.ENCODER.hidden_size,
                   enc_recurrent=cfg.MODEL.ENCODER.recurrent,
                   enc_activation=cfg.MODEL.ENCODER.activation,
                   enc_pre_norm=cfg.MODEL.ENCODER.pre_norm,
                   dec_num_layers=cfg.MODEL.DECODER.num_layers,
                   dec_units=cfg.MODEL.DECODER.units,
                   dec_num_heads=cfg.MODEL.DECODER.num_heads,
                   dec_hidden_size=cfg.MODEL.DECODER.hidden_size,
                   dec_recurrent=cfg.MODEL.DECODER.recurrent,
                   dec_activation=cfg.MODEL.DECODER.activation,
                   dec_pre_norm=cfg.MODEL.DECODER.pre_norm,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   dtype=cfg.MODEL.dtype)

BACKBONE_REGISTRY.register('bart', [BartModel,
                                    get_pretrained_bart,
                                    list_pretrained_bart])
