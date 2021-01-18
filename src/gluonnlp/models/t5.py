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


__all__ = ['T5Model', 'T5Inference']


import os
import functools
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet import np, npx
from mxnet.gluon import HybridBlock, Parameter, nn
from mxnet.initializer import Constant, Normal, Xavier
from ..attention_cell import (
    gen_self_attn_mask, gen_mem_attn_mask, MultiHeadAttentionCell, RelAttentionScoreCell
)
from .base import BACKBONE_REGISTRY
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..data import Vocab
from ..data.tokenizers import SentencepieceTokenizer
from ..layers import RMSNorm, PositionwiseFFN
from ..sequence_sampler import BaseStepDecoder
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..utils.registry import Registry


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 't5.txt'))

t5_cfg_reg = Registry('t5_cfg')


@t5_cfg_reg.register()
def google_t5_base(): 
    """Configuratino of T5 Base"""
    cfg = CN()
    # model parameters
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 32128
    cfg.MODEL.d_model = 768
    cfg.MODEL.d_kv = 64
    cfg.MODEL.d_ff = 3072
    cfg.MODEL.num_layers = 12
    cfg.MODEL.num_heads = 12
    cfg.MODEL.dropout_prob = 0.1
    cfg.MODEL.layer_norm_eps = 1E-6
    cfg.MODEL.activation = 'relu'
    cfg.MODEL.dtype = 'float32'
    cfg.MODEL.layout = 'NT'
    # initializer parameters
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.init_factor = 1.0
    # other parameters
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@t5_cfg_reg.register()
def google_t5_small(): 
    cfg = google_t5_base()
    cfg.defrost()
    cfg.MODEL.d_model = 512
    cfg.MODEL.d_ff = 2048
    cfg.MODEL.num_layers = 6
    cfg.MODEL.num_heads = 8
    cfg.freeze()
    return cfg


@t5_cfg_reg.register()
def google_t5_large(): 
    cfg = google_t5_base()
    cfg.defrost()
    cfg.MODEL.d_model = 1024
    cfg.MODEL.d_ff = 4096
    cfg.MODEL.num_layers = 24
    cfg.MODEL.num_heads = 16
    cfg.freeze()
    return cfg


@t5_cfg_reg.register()
def google_t5_3B(): 
    cfg = google_t5_base()
    cfg.defrost()
    cfg.MODEL.d_model = 1024
    cfg.MODEL.d_kv = 128
    cfg.MODEL.d_ff = 16384
    cfg.MODEL.num_layers = 24
    cfg.MODEL.num_heads = 32
    cfg.freeze()
    return cfg


@t5_cfg_reg.register()
def google_t5_11B(): 
    cfg = google_t5_base()
    cfg.defrost()
    cfg.MODEL.d_model = 1024
    cfg.MODEL.d_kv = 128
    cfg.MODEL.d_ff = 65536
    cfg.MODEL.num_layers = 24
    cfg.MODEL.num_heads = 128
    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'google_t5_small': {
        'cfg': google_t5_small(), 
        'vocab': 'google_t5_small/t5-5f05e7c5.vocab', 
        'params': 'google_t5_small/model-e34f6fbd.params'
    }, 
    'google_t5_base': {
        'cfg': google_t5_base(), 
        'vocab': 'google_t5_base/t5-5f05e7c5.vocab', 
        'params': 'google_t5_base/model-e1956ac9.params'
    }, 
    'google_t5_large': {
        'cfg': google_t5_large(), 
        'vocab': 'google_t5_large/t5-5f05e7c5.vocab', 
        'params': 'google_t5_large/model-bf5fc813.params'
    }, 
    'google_t5_3B': {
        'cfg': google_t5_3B()
    }, 
    'google_t5_11B': {
        'cfg': google_t5_11B()
    }
}


@use_np
class T5Block(HybridBlock): 
    def __init__(
        self, 
        d_model, 
        d_kv, 
        d_ff, 
        is_decoder, 
        num_heads=12, 
        dropout_prob=0.1, 
        layer_norm_eps=1E-6, 
        activation='relu', 
        init_factor=1.0, 
        layout='NT', 
        dtype='float32'
    ): 
        super().__init__()
        self._d_model = d_model
        self._d_kv = d_kv
        self._d_ff = d_ff
        self._is_decoder = is_decoder
        self._num_heads = num_heads
        self._inner_dim = self._num_heads * self._d_kv
        self._dtype = dtype
        assert layout in ['TN', 'NT'], \
            'Invalid layout: {}. Only "TN" and "NT" are supported.'.format(layout)
        self._layout = layout

        self.self_attn_layer_norm = RMSNorm(
            in_channels=d_model, 
            center=False, 
            scale=True, 
            gamma_initializer=Constant(1.0 * init_factor), 
            variance_epsilon=layer_norm_eps, 
            dtype=dtype
        )
        # avoid scaling before softmax
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
        self.self_attn_q = nn.Dense(
                units=self._inner_dim, 
                in_units=d_model, 
                flatten=False, 
                use_bias=False, 
                weight_initializer=Normal((d_model * d_kv) ** -0.5 * init_factor), 
                dtype=dtype
            )
        self.self_attn_k = nn.Dense(
            units=self._inner_dim, 
            in_units=d_model, 
            flatten=False, 
            use_bias=False, 
            weight_initializer=Normal(d_model ** -0.5 * init_factor), 
            dtype=dtype
        )
        self.self_attn_v = nn.Dense(
            units=self._inner_dim, 
            in_units=d_model, 
            flatten=False, 
            use_bias=False, 
            weight_initializer=Normal(d_model ** -0.5 * init_factor), 
            dtype=dtype
        )
        self.self_attn = MultiHeadAttentionCell(
            query_units=self._inner_dim, 
            num_heads=num_heads, 
            attention_dropout=dropout_prob, 
            scaled=False, 
            normalized=False, 
            dtype=dtype, 
            layout='NTK' if layout == 'NT' else 'TNK', 
            use_einsum=False
        )
        self.self_attn_proj = nn.Dense(
            units=d_model, 
            in_units=self._inner_dim, 
            flatten=False, 
            use_bias=False, 
            weight_initializer=Normal(self._inner_dim ** -0.5 * init_factor), 
            dtype=dtype
        )
        if is_decoder: 
            self.cross_attn_layer_norm = RMSNorm(
                in_channels=d_model, 
                center=False, 
                scale=True, 
                gamma_initializer=Constant(1.0 * init_factor), 
                variance_epsilon=layer_norm_eps, 
                dtype=dtype
            )
            # avoid scaling before softmax
            self.cross_attn_q = nn.Dense(
                units=self._inner_dim, 
                in_units=d_model, 
                flatten=False, 
                use_bias=False, 
                weight_initializer=Normal((d_model * d_kv) ** -0.5 * init_factor), 
                dtype=dtype
            )
            self.cross_attn_k = nn.Dense(
                units=self._inner_dim, 
                in_units=d_model, 
                flatten=False, 
                use_bias=False, 
                weight_initializer=Normal(d_model ** -0.5 * init_factor), 
                dtype=dtype
            )
            self.cross_attn_v = nn.Dense(
                units=self._inner_dim, 
                in_units=d_model, 
                flatten=False, 
                use_bias=False, 
                weight_initializer=Normal(d_model ** -0.5 * init_factor), 
                dtype=dtype
            )
            self.cross_attn = MultiHeadAttentionCell(
                query_units=self._inner_dim, 
                num_heads=num_heads, 
                attention_dropout=dropout_prob, 
                scaled=False, 
                normalized=False, 
                dtype=dtype, 
                layout='NTK' if layout == 'NT' else 'TNK', 
                use_einsum=False
            )
            self.cross_attn_proj = nn.Dense(
                units=d_model, 
                in_units=self._inner_dim, 
                flatten=False, 
                use_bias=False, 
                weight_initializer=Normal(self._inner_dim ** -0.5 * init_factor), 
                dtype=dtype
            )
        assert activation in ['relu', 'gated-gelu'], \
            '{} is not supported. Please choose from "relu" and "gated-gelu"'.format(activation)
        # the weight_initializer here is equivalent to Normal(in_units ** -0.5 * init_factor)
        self.ffn = PositionwiseFFN(
            units=d_model, 
            hidden_size=d_ff, 
            use_bias=False, 
            activation_dropout=dropout_prob, 
            dropout=dropout_prob, 
            weight_initializer=Xavier('gaussian', 'in', np.sqrt(init_factor)),  
            activation='relu' if activation == 'relu' else 'gelu(tanh)', 
            use_gated_activation=False if activation == 'relu' else True, 
            normalization='rms_norm', 
            layer_norm_eps=layer_norm_eps, 
            pre_norm=True, 
            dtype=dtype, 
            center=False, 
            scale=True, 
            gamma_initializer=Constant(1.0 * init_factor)
        )
        self.dropout = nn.Dropout(dropout_prob)

    def _assert_decoder_method(fn): 
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs): 
            assert self._is_decoder, \
                '{}() is available for decoder only.'.format(fn.__name__)
            return fn(self *args, **kwargs)
        return wrapper

    @property
    def layout(self): 
        return self._layout

    @_assert_decoder_method
    @property
    def state_batch_axis(self): 
        if self.layout == 'NT': 
            return 0, 0
        else: 
            return 1, 1

    @_assert_decoder_method
    def init_states(self, batch_size, ctx, dtype='float32'): 
        if self.layout == 'NT': 
            shape = (batch_size, 0, self._num_heads, self._d_kv)
        else: 
            shape = (0, batch_size, self._num_heads, self._d_kv)
        init_key = np.zeros(shape, ctx=ctx, dtype=dtype)
        init_value = np.zeros(shape, ctx=ctx, dtype=dtype)
        return init_key, init_value

    @_assert_decoder_method
    def incremental_decode(
        self, 
        hidden_states, 
        past_key_value, 
        mem_states, 
        mem_valid_length, 
        mem_attn_mask=None
    ): 
        raise NotImplementedError

    def forward(
        self, 
        hidden_states, 
        self_attn_mask, 
        position_embeddings, 
        mem_states=None, 
        mem_attn_mask=None, 
        mem_position_embeddings=None
    ): 
        """
        hidden_states: 
            - layout = 'NT'
                Shape (B, L_seq, d_model)
            - layout = 'TN'
                Shape (L_seq, B, d_model)
        """
        # NT -> NTK: (B, L_seq, inner_dim) -> (B, L_seq, num_heads, n_kv)
        # TN -> TNK: (L_seq, B, inner_dim) -> (L_seq, B, num_heads, n_kv)
        def transpose_for_scores(x):
            return npx.reshape(x, (-2, -2, self._num_heads, -1))

        # 1. self-attention
        out = self.self_attn_layer_norm(hidden_states)
        self_query, self_key, self_value = (
            self.self_attn_q(out), 
            self.self_attn_k(out), 
            self.self_attn_v(out)
        )
        out, _ = self.self_attn(
            transpose_for_scores(self_query), 
            transpose_for_scores(self_key), 
            transpose_for_scores(self_value), 
            self_attn_mask, 
            position_embeddings
        )
        out = self.dropout(self.self_attn_proj(out))
        hidden_states = hidden_states + out

        # 2. cross-attention, if needed
        if self._is_decoder: 
            out = self.cross_attn_layer_norm(hidden_states)
            cross_query, cross_key, cross_value = (
                self.cross_attn_q(out), 
                self.cross_attn_k(mem_states), 
                self.cross_attn_v(mem_states)
            )
            out, _ = self.cross_attn(
                transpose_for_scores(cross_query), 
                transpose_for_scores(cross_key), 
                transpose_for_scores(cross_value), 
                mem_attn_mask, 
                mem_position_embeddings
            )
            out = self.dropout(self.cross_attn_proj(out))
            hidden_states = hidden_states + out

        # 3. feed forward
        hidden_states = self.ffn(hidden_states)
        return hidden_states


@use_np
class T5Encoder(HybridBlock): 
    def __init__(
        self, 
        d_model, 
        d_kv, 
        d_ff, 
        num_layers=12, 
        num_heads=12, 
        dropout_prob=0.1, 
        layer_norm_eps=1E-6, 
        activation='relu', 
        init_factor=1.0, 
        layout='NT', 
        dtype='float32'
    ): 
        super().__init__()
        self._d_model = d_model
        self._d_kv = d_kv
        self._d_ff = d_ff
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._inner_dim = num_heads * d_kv
        self._dtype = dtype
        assert layout in ['TN', 'NT'], \
            'Invalid layout: {}. Only "TN" and "NT" are supported.'.format(layout)
        self._layout = layout
        self.time_axis = 1 if self.layout == 'NT' else 0

        self.relative_position_encoder = RelAttentionScoreCell(
            query_units=self._inner_dim, 
            num_heads=num_heads, 
            method='t5', 
            bidirectional=True, 
            embed_initializer=Normal(d_model ** -0.5 * init_factor), 
            layout='NTK' if layout == 'NT' else 'TNK', 
            dtype=dtype
        )
        self.layers = nn.HybridSequential()
        for _ in range(num_layers): 
            self.layers.add(
                T5Block(
                    d_model=d_model, 
                    d_kv=d_kv, 
                    d_ff=d_ff, 
                    is_decoder=False, 
                    num_heads=num_heads, 
                    dropout_prob=dropout_prob, 
                    layer_norm_eps=layer_norm_eps, 
                    activation=activation, 
                    init_factor=init_factor, 
                    layout=layout, 
                    dtype=dtype
                )
            )
        self.final_layer_norm = RMSNorm(
            in_channels=d_model, 
            center=False, 
            scale=True, 
            gamma_initializer=Constant(1.0 * init_factor), 
            variance_epsilon=layer_norm_eps, 
            dtype=dtype
        )
        self.dropout = nn.Dropout(dropout_prob)

    @property
    def layout(self): 
        return self._layout

    def _get_relative_position(self, hidden_states): 
        query_position = np.expand_dims(
            npx.arange_like(hidden_states, axis=self.time_axis), 
            axis=-1
        )
        mem_position = np.expand_dims(
            npx.arange_like(hidden_states, axis=self.time_axis), 
            axis=0
        )
        relative_position = mem_position - query_position
        return relative_position.astype(np.int32)

    def forward(self, hidden_states, valid_length): 
        # 1. relative position embeddings and attention masks
        position_embeddings = self.relative_position_encoder(
            self._get_relative_position(hidden_states)
        )
        self_attn_mask = gen_self_attn_mask(
            hidden_states, 
            valid_length, 
            dtype=self._dtype, 
            attn_type='full', 
            layout=self.layout
        )

        # 2. encoder blocks and other layers
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers: 
            hidden_states = layer(
                hidden_states, 
                self_attn_mask, 
                position_embeddings
            )
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


@use_np
class T5Decoder(HybridBlock): 
    def __init__(
        self, 
        d_model, 
        d_kv, 
        d_ff, 
        num_layers=12, 
        num_heads=12, 
        dropout_prob=0.1, 
        layer_norm_eps=1E-6, 
        activation='relu', 
        init_factor=1.0, 
        layout='NT', 
        dtype='float32'
    ): 
        super().__init__()
        self._d_model = d_model
        self._d_kv = d_kv
        self._d_ff = d_ff
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._inner_dim = num_heads * d_kv
        self._dtype = dtype
        assert layout in ['TN', 'NT'], \
            'Invalid layout: {}. Only "TN" and "NT" are supported.'.format(layout)
        self._layout = layout
        self.time_axis = 1 if self.layout == 'NT' else 0

        self.relative_position_encoder = RelAttentionScoreCell(
            query_units=self._inner_dim, 
            num_heads=num_heads, 
            method='t5', 
            bidirectional=False, 
            embed_initializer=Normal(d_model ** -0.5 * init_factor), 
            layout='NTK' if layout == 'NT' else 'TNK', 
            dtype=dtype
        )
        self.layers = nn.HybridSequential()
        for _ in range(num_layers): 
            self.layers.add(
                T5Block(
                    d_model=d_model, 
                    d_kv=d_kv, 
                    d_ff=d_ff, 
                    is_decoder=True, 
                    num_heads=num_heads, 
                    dropout_prob=dropout_prob, 
                    layer_norm_eps=layer_norm_eps, 
                    activation=activation, 
                    init_factor=init_factor, 
                    layout=layout, 
                    dtype=dtype
                )
            )
        self.final_layer_norm = RMSNorm(
            in_channels=d_model, 
            center=False, 
            scale=True, 
            gamma_initializer=Constant(1.0 * init_factor), 
            variance_epsilon=layer_norm_eps, 
            dtype=dtype
        )
        self.dropout = nn.Dropout(dropout_prob)

    @property
    def layout(self): 
        return self._layout

    @property
    def state_batch_axis(self): 
        return list(layer.state_batch_axis for layer in self.layers)

    def init_states(self, batch_size, ctx, dtype='float32'): 
        return list(layer.init_states(batch_size, ctx, dtype) for layer in self.layers)

    def incremental_decode(
        hidden_states, 
        past_key_value, 
        mem_states, 
        mem_valid_length
    ): 
        raise NotImplementedError

    def _get_relative_position(self, hidden_states, mem_states=None, past_key_value=None): 
        if past_key_value is None: 
            query_position = np.expand_dims(
                npx.arange_like(hidden_states, axis=self.time_axis), 
                axis=-1
            )
        else: 
            # for incremental decoding only, where past key and past value are of shape
            # NT(NTK): (B, L_seq, num_heads, n_kv); TN(TNK): (L_seq, B, num_heads, n_kv)
            query_position = npx.arange_like(
                np.concatenate([hidden_states, past_key_value[0]], axis=self.time_axis), 
                axis=self.time_axis
            )
            query_position = np.expand_dims(query_position, axis=-1)
        mem_position = np.expand_dims(
            npx.arange_like(hidden_states if mem_states is None else mem_states, axis=self.time_axis), 
            axis=0
        )
        relative_position = mem_position - query_position
        return relative_position.astype(np.int32)

    def forward(self, hidden_states, valid_length, mem_states, mem_valid_length): 
        # 1. relative position embeddings and attention masks
        position_embeddings = self.relative_position_encoder(
            self._get_relative_position(hidden_states)
        )
        # relative position embedding is not used for cross attention, 
        # so we just obtain the correct shape and fill it with 0
        mem_relative_position = np.zeros_like(
            self._get_relative_position(hidden_states, mem_states)
        )
        mem_position_embeddings = np.repeat(
            np.expand_dims(mem_relative_position, axis=0), 
            self._num_heads, 
            axis=0
        )
        self_attn_mask = gen_self_attn_mask(
            hidden_states, 
            valid_length, 
            dtype=self._dtype, 
            attn_type='causal', 
            layout=self.layout
        )
        mem_attn_mask = gen_mem_attn_mask(
            mem_states, 
            mem_valid_length, 
            hidden_states, 
            valid_length, 
            dtype=self._dtype, 
            layout=self.layout
        )

        # 2. decoder blocks and other layers
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers: 
            hidden_states = layer(
                hidden_states, 
                self_attn_mask, 
                position_embeddings, 
                mem_states, 
                mem_attn_mask, 
                mem_position_embeddings
            )
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


@use_np
class T5Model(HybridBlock): 
    def __init__(
        self, 
        vocab_size=32128, 
        d_model=768, 
        d_kv=64, 
        d_ff=3072, 
        num_layers=12, 
        num_heads=12, 
        dropout_prob=0.1, 
        layer_norm_eps=1E-6, 
        activation='relu', 
        init_factor=1.0, 
        layout='NT', 
        dtype='float32'
    ): 
        super().__init__()
        assert vocab_size > 0, 'Vocab size {} is not valid.'.format(vocab_size)
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        self._activation = activation
        assert layout in ['TN', 'NT'], \
            'Invalid layout: {}. Only "TN" and "NT" are supported.'.format(layout)
        self._layout = layout

        # input embedding weights are shared between across encoder and decoder
        self.input_embedding_layer = nn.Embedding(
            input_dim=vocab_size, 
            output_dim=d_model, 
            weight_initializer=Normal(1.0 * init_factor), 
            dtype=dtype
        )
        self.encoder = T5Encoder(
            d_model=d_model, 
            d_kv=d_kv, 
            d_ff=d_ff, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            dropout_prob=dropout_prob, 
            layer_norm_eps=layer_norm_eps, 
            activation=activation,  
            init_factor=init_factor, 
            layout=layout, 
            dtype=dtype
        )
        self.decoder = T5Decoder(
            d_model=d_model, 
            d_kv=d_kv, 
            d_ff=d_ff, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            dropout_prob=dropout_prob, 
            layer_norm_eps=layer_norm_eps, 
            activation=activation,  
            init_factor=init_factor, 
            layout=layout, 
            dtype=dtype
        )
    
    @property
    def activation(self): 
        return self._activation
    
    @property
    def layout(self): 
        return self._layout

    @property
    def num_layers(self): 
        return self._num_layers

    @property
    def vocab_size(self): 
        return self._vocab_size

    def forward(self, src_data, src_valid_length, tgt_data, tgt_valid_length): 
        src_hidden_states = self.input_embedding_layer(src_data)
        enc_out = self.encoder(
            src_hidden_states, 
            src_valid_length
        )
        tgt_hidden_states = self.input_embedding_layer(tgt_data)
        dec_out = self.decoder(
            tgt_hidden_states, 
            tgt_valid_length, 
            enc_out, 
            src_valid_length
        )
        return dec_out

    @classmethod
    def get_cfg(cls, key=None): 
        if key is None: 
            return google_t5_base()
        else: 
            return t5_cfg_reg.create(key)

    @classmethod
    def from_cfg(cls, cfg, dtype=None): 
        cfg = cls.get_cfg().clone_merge(cfg)
        assert cfg.VERSION == 1, 'Wrong version: {}.'.format(cfg.VERSION)
        if dtype is None: 
            dtype = cfg.MODEL.dtype
        return cls(
            vocab_size=cfg.MODEL.vocab_size, 
            d_model=cfg.MODEL.d_model, 
            d_kv=cfg.MODEL.d_kv, 
            d_ff=cfg.MODEL.d_ff, 
            num_layers=cfg.MODEL.num_layers, 
            num_heads=cfg.MODEL.num_heads, 
            dropout_prob=cfg.MODEL.dropout_prob, 
            layer_norm_eps=cfg.MODEL.layer_norm_eps, 
            activation=cfg.MODEL.activation, 
            init_factor=cfg.INITIALIZER.init_factor, 
            layout=cfg.MODEL.layout, 
            dtype=dtype
        )


@use_np
class T5Inference(HybridBlock, BaseStepDecoder): 
    pass 


def list_pretrained_t5(): 
    return sorted(list(PRETRAINED_URL.keys()))


def build_t5_tokenizer(vocab_path, do_lower, extra_ids=100): 
    # extend tokens in vocab with <extra_id>s, which correspond to noise span sentinels one-by-one
    # <extra_id_0> will the last token in the new vocabulary
    extra_token = '<extra_id_{}>'
    special_tokens = {
        'extra{}_token'.format(i): extra_token.format(i) for i in range(extra_ids - 1, -1, -1)
    }
    spiece_model = SentencepieceTokenizer(vocab_path)
    tokens = spiece_model.vocab.all_tokens
    tokens.extend(list(special_tokens.values()))
    # re-specify special tokens 
    special_tokens['eos_token'] = spiece_model.vocab.eos_token
    special_tokens['unk_token'] = spiece_model.vocab.unk_token
    special_tokens['pad_token'] = spiece_model.vocab.pad_token
    tokenizer = SentencepieceTokenizer(
        model_path=vocab_path, 
        vocab=Vocab(tokens, **special_tokens), 
        lowercase=do_lower
    )
    # sanity check: every additional token has been inserted with correct order
    inserted_special_tokens = list(extra_token.format(i) for i in range(extra_ids - 1, -1, -1))
    assert list(
        tokenizer.vocab.to_tokens(i) for i in range(len(tokenizer._sp_model), len(tokenizer._vocab))
    ) == inserted_special_tokens, 'Some <extra_id> tokens are not properly inserted.'
    return tokenizer


def get_pretrained_t5(
    model_name: str = 't5-base', 
    root: str = get_model_zoo_home_dir(), 
    load_backbone: bool = True, 
    load_lm: bool = False, 
    extra_ids: int = 100
) -> Tuple[CN, SentencepieceTokenizer, str, str]: 
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}.'.format(
        model_name, list_pretrained_t5())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']

    local_paths = dict()
    download_jobs = [('vocab', vocab_path)]
    if cfg is None: 
        download_jobs.append(('cfg', cfg_path))
    for key, path in download_jobs: 
        local_paths[key] = download(
            url=get_repo_model_zoo_url() + path, 
            path=os.path.join(root, path), 
            sha1_hash=FILE_STATS[path]
        )
    if load_backbone: 
        local_params_path = download(
            url=get_repo_model_zoo_url() + params_path, 
            path=os.path.join(root, params_path), 
            sha1_hash=FILE_STATS[params_path]
        )
    else: 
        local_params_path = None
    # lm model has not been implemented
    local_lm_params_path = None
    tokenizer = build_t5_tokenizer(vocab_path, False, extra_ids)
    if cfg is None: 
        cfg = T5Model.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_lm_params_path


BACKBONE_REGISTRY.register(
    't5', 
    [T5Model, get_pretrained_t5, list_pretrained_t5]
)
