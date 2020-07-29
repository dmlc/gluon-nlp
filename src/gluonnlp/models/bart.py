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
from .transformer import TransformerModel
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..initializer import TruncNorm
from ..attention_cell import MultiHeadAttentionCell, gen_self_attn_mask
from ..layers import get_activation, PositionalEmbedding, PositionwiseFFN, InitializerType
from ..op import select_vectors_by_position
from ..utils.registry import Registry
from ..data.tokenizers import HuggingFaceWordPieceTokenizer

bart_cfg_reg = Registry('bart_cfg')


@bart_cfg_reg.register()
def bart_base():
    cfg = CN()
    # Config for the bart base model
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 51201
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.layernorm_embedding = True
    cfg.MODEL.scale_embed = False
    cfg.MODEL.shared_embed = True
    cfg.MODEL.tie_weights = True
    cfg.MODEL.cross_self_attention = False
    cfg.MODEL.attention_dropout_prob = 0.0
    cfg.MODEL.activation_dropout = 0.0
    cfg.MODEL.hidden_dropout_prob = 0.1
    cfg.MODEL.layer_norm_eps = 1E-5
    cfg.MODEL.pooler_activation = 'tanh'
    cfg.MODEL.dtype = 'float32'

    # Parameters for the encoder
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.max_length = 1024
    cfg.MODEL.ENCODER.num_layers = 6
    cfg.MODEL.ENCODER.units = 768
    cfg.MODEL.ENCODER.num_heads = 12
    cfg.MODEL.ENCODER.hidden_size = 3072
    cfg.MODEL.ENCODER.recurrent = False
    cfg.MODEL.ENCODER.pre_norm = False
    cfg.MODEL.ENCODER.activation = 'gelu'

    # Parameters for the decoder
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.max_length = 1024
    cfg.MODEL.DECODER.num_layers = 6
    cfg.MODEL.DECODER.units = 768
    cfg.MODEL.DECODER.num_heads = 12
    cfg.MODEL.DECODER.hidden_size = 3072
    cfg.MODEL.DECODER.recurrent = False
    cfg.MODEL.DECODER.pre_norm = False
    cfg.MODEL.DECODER.activation = 'gelu'

    # Parameters for the initializer
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['normal', 768**-0.5]
    cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 1.0]
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

    cfg.INITIALIZER.embed = ['normal', 0.03125]
    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'fairseq_bart_base': {
        'cfg': bart_base(),
        'merges': 'fairseq_bart_base/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_bart_base/',
        'params': 'fairseq_bart_base/',
        'lowercase': False,
    },
    'fairseq_bart_large': {
        'cfg': bart_large(),
        'merges': 'fairseq_bart_base/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_bart_large/',
        'params': 'fairseq_bart_large/',
        'lowercase': False,
    }
}


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'bart.txt'))


@use_np
class BartModel(TransformerModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert  self._src_vocab_size == self._tgt_vocab_size, 'Vocab size mismatch between encoder and decoder'
        self._vocab_size = self._src_vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size

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
                   max_src_length=cfg.MODEL.max_src_length,
                   max_tgt_length=cfg.MODEL.max_tgt_length,
                   scale_embed=cfg.MODEL.scale_embed,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   layernorm_embedding=cfg.MODEL.layernorm_embedding,
                   shared_embed=cfg.MODEL.shared_embed,
                   tie_weights=cfg.MODEL.tie_weights,
                   attention_dropout=cfg.MODEL.attention_dropout_prob,
                   activation_dropout=cfg.MODEL.activation_dropout,
                   dropout=cfg.MODEL.hidden_dropout_prob,
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

# BACKBONE_REGISTRY.register('bart', [BartModel,
        # get_pretrained_bart,
        # list_pretrained_bart])
