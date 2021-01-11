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
T5 Model

@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}
"""

__all__ = []

import os
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet import np, npx
from mxnet.gluon import HybridBlock, Parameter, nn
from ..base import get_model_zoo_home_dir, get_model_zoo_checksum_dir
from ..data import Vocab
from ..data.tokenizers import SentencepieceTokenizer
from ..layers import get_activation
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats
from ..utils.registry import Registry

t5_cfg_reg = Registry('t5_cfg')


@t5_cfg_reg.register()
def google_t5_base(): 
    cfg = CN()
    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'google_t5-small': {

    }, 
    'google_t5-base': {

    }, 
    'google_t5-large': {

    }, 
    'google_t5: 3b': {

    }, 
    'google_t5: 11b': {

    }
}

# FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 't5.txt'))


@use_np
class T5LayerNorm(HybridBlock): 
    def __init__(self, d_model, eps=1e-6): 
        """
        LayerNorm with no bias or mean substraction
        """
        super().__init__()
        self.gemma = Parameter('layernorm_weight', shape=d_model, init='ones')
        self.variance_epsilon = eps

    def forward(self, x): 
        var = np.power(x.astype('float32'), 2).mean(-1, keepdims=True)
        x = x * np.reciprocal(np.sqrt(var + self.variance_epsilon))
        if self.gemma.dtype == 'float16': 
            x = x.astype('float16')
        return self.gemma * x


@use_np
class T5DenseReluDense(HybridBlock): 
    def __init__(self, d_model, d_ff, dropout_prob): 
        super.__init__()
        self.wi = nn.Dense(units=d_ff, in_units=d_model, use_bias=False)
        self.relu = get_activation('relu')
        self.wo = nn.Dense(units=d_model, in_units=d_ff, use_bias=False)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x): 
        x = self.wi(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.wo(x)
        return x


@use_np
class T5DenseGatedGeluDense(HybridBlock): 
    def __init__(self, d_model, d_ff, dropout_prob): 
        self.wi_0 = nn.Dense(units=d_ff, in_units=d_model, use_bias=False)
        self.wi_1 = nn.Dense(units=d_ff, in_units=d_model, use_bias=False)
        self.wo = nn.Dense(units=d_model, in_units=d_ff, use_bias=False)
        self.gelu = get_activation('gelu')
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x): 
        x_gelu = self.gelu(self.wi_0(x))
        x_linear = self.wi_1(x)
        x = x_gelu * x_linear
        x = self.dropout(x)
        x = self.wo(x)
        return x


@use_np
class T5BlockFFN(HybridBlock): 
    def __init__(
        self, 
        d_model, 
        d_ff, 
        dropout_prob, 
        ff_proj, 
        layer_norm_eps
    ): 
        super().__init__()
        if ff_proj == 'relu': 
            self.ffn = T5DenseReluDense(d_model, d_ff, dropout_prob)
        elif ff_proj == 'gated-gelu': 
            self.ffn = T5DenseGatedGeluDense(d_model, d_ff, dropout_prob)
        else: 
            raise ValueError(
                '{} unsupported. Select `relu` or `gated-gelu`'.format(ff_proj)
            )
        self.layer_norm = T5LayerNorm(d_model, layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x): 
        out = self.layer_norm(x)
        out = self.ffn(out)
        out = x + self.dropout(out)
        return out


@use_np
class T5Attention(HybridBlock): 
    pass


@use_np
class T5Block(HybridBlock): 
    pass


@use_np
class T5Model(HybridBlock): 
    pass


def list_pretrained_t5(): 
    return sorted(list(PRETRAINED_URL.keys()))


def _build_t5_tokenizer(vocab_path, do_lower, extra_ids): 
    # manually add additional special tokens corresponding to noise span sentinels
    # with <extra_id_0> be the last token in the new vocabulary
    extra_token = '<extra_id_{}>'
    additional_special_tokens = {
        'extra{}_token'.format(i): extra_token.format(i) for i in range(extra_ids - 1, -1, -1)
    }
    tokenizer = SentencepieceTokenizer(
        model_path=vocab_path,
        lowercase=do_lower, 
        **additional_special_tokens
    )
    # sanity check: every additional token has been inserted with correct order
    inserted_special_tokens = list(extra_token.format(i) for i in range(extra_ids - 1, -1, -1))
    assert list(
        tokenizer._vocab.to_tokens(i) for i in range(len(tokenizer._sp_model), len(tokenizer._vocab))
    ) == inserted_special_tokens, 'Some <extra_id> tokens are not properly inserted'
    return tokenizer


def get_pretrained_t5(model_name: str = 't5-base', 
                      root: str = get_model_zoo_home_dir(), 
                      load_backbone: bool = True, 
                      load_lm: bool = False, 
                      extra_ids: int = 100) \
    -> Tuple[CN, SentencepieceTokenizer, str, str]: 
    """
    TBD
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_t5())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None

    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']

    do_lower = True if 'lowercase' in PRETRAINED_URL[model_name]\
                       and PRETRAINED_URL[model_name]['lowercase'] else False
    tokenizer = _build_t5_tokenizer(vocab_path, do_lower, extra_ids)
    if cfg is None: 
        cfg = T5Model.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer,
