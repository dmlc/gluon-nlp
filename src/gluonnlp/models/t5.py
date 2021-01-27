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


__all__ = ['T5Model', 'T5Inference', 'T5Tokenizer', 'T5NMTInference']


import os
import functools
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet import np, npx
from mxnet.gluon import HybridBlock, Parameter, nn
from mxnet.initializer import Constant, Normal, Xavier
import numpy as _np
from ..attention_cell import (
    gen_self_attn_mask, gen_mem_attn_mask, MultiHeadAttentionCell, gen_rel_position, RelAttentionScoreCell
)
from .base import BACKBONE_REGISTRY
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..data import Vocab
from ..data.tokenizers import SentencepieceTokenizer
from ..data.tokenizers.base import is_tokens_from_multiple_sentences, get_token_type
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
        'cfg': google_t5_3B(), 
        'vocab': 'google_t5_3B/t5-5f05e7c5.vocab', 
        'params': 'google_t5_3B/model-48ba7250.params'
    }, 
    'google_t5_11B': {
        'cfg': google_t5_11B(), 
        'vocab': 'google_t5_11B/t5-5f05e7c5.vocab', 
        'params': 'google_t5_11B/model-1936031c.params'
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
        self._time_axis = 1 if self.layout == 'NT' else 0

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
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    def layout(self): 
        return self._layout

    @property
    @_assert_decoder_method
    def state_batch_axis(self): 
        if self.layout == 'NT': 
            return 0, 0
        else: 
            return 1, 1

    @_assert_decoder_method
    def _init_key_value(self, batch_size, ctx, dtype='float32'): 
        if self.layout == 'NT': 
            shape = (batch_size, 0, self._num_heads, self._d_kv)
        else: 
            shape = (0, batch_size, self._num_heads, self._d_kv)
        init_key = np.zeros(shape, ctx=ctx, dtype=dtype)
        init_value = np.zeros(shape, ctx=ctx, dtype=dtype)
        return init_key, init_value

    def transpose_for_scores(self, x): 
        # NT -> NTK: (B, L_seq, inner_dim) -> (B, L_seq, num_heads, n_kv)
        # TN -> TNK: (L_seq, B, inner_dim) -> (L_seq, B, num_heads, n_kv)
        return npx.reshape(x, (-2, -2, self._num_heads, -1))

    @_assert_decoder_method
    def incremental_decode(
        self, 
        step_hidden_states, 
        step_position_embeddings, 
        past_key_value, 
        mem_states, 
        step_mem_attn_mask
    ): 
        # 1. self-attention
        out = self.self_attn_layer_norm(step_hidden_states)
        step_self_query, step_self_key, step_self_value = (
            self.transpose_for_scores(self.self_attn_q(out)), 
            self.transpose_for_scores(self.self_attn_k(out)), 
            self.transpose_for_scores(self.self_attn_v(out))
        )
        self_key, self_value = (
            np.concatenate([past_key_value[0], step_self_key], axis=self._time_axis), 
            np.concatenate([past_key_value[1], step_self_value], axis=self._time_axis)
        )
        out, _ = self.self_attn(
            step_self_query, 
            self_key, 
            self_value, 
            None, 
            step_position_embeddings
        )
        out = self.dropout(self.self_attn_proj(out))
        step_hidden_states = step_hidden_states + out

        # 2. cross-attention
        out = self.cross_attn_layer_norm(step_hidden_states)
        step_cross_query, cross_key, cross_value = (
            self.transpose_for_scores(self.cross_attn_q(out)), 
            self.transpose_for_scores(self.cross_attn_k(mem_states)), 
            self.transpose_for_scores(self.cross_attn_v(mem_states))
        )
        out, _ = self.cross_attn(
            step_cross_query, 
            cross_key, 
            cross_value, 
            step_mem_attn_mask
        )
        out = self.dropout(self.cross_attn_proj(out))
        step_hidden_states = step_hidden_states + out

        # 3. feed forward
        step_hidden_states = self.ffn(step_hidden_states)
        return step_hidden_states, (self_key, self_value)

    def forward(
        self, 
        hidden_states, 
        self_attn_mask, 
        position_embeddings, 
        mem_states=None, 
        mem_attn_mask=None
    ): 
        """
        hidden_states: 
            - layout = 'NT'
                Shape (B, L_seq, d_model)
            - layout = 'TN'
                Shape (L_seq, B, d_model)
        """
        # 1. self-attention
        out = self.self_attn_layer_norm(hidden_states)
        self_query, self_key, self_value = (
            self.transpose_for_scores(self.self_attn_q(out)), 
            self.transpose_for_scores(self.self_attn_k(out)), 
            self.transpose_for_scores(self.self_attn_v(out))
        )
        out, _ = self.self_attn(
            self_query, 
            self_key, 
            self_value, 
            self_attn_mask, 
            position_embeddings
        )
        out = self.dropout(self.self_attn_proj(out))
        hidden_states = hidden_states + out

        # 2. cross-attention, if needed
        if self._is_decoder: 
            out = self.cross_attn_layer_norm(hidden_states)
            cross_query, cross_key, cross_value = (
                self.transpose_for_scores(self.cross_attn_q(out)), 
                self.transpose_for_scores(self.cross_attn_k(mem_states)), 
                self.transpose_for_scores(self.cross_attn_v(mem_states))
            )
            out, _ = self.cross_attn(
                cross_query, 
                cross_key, 
                cross_value, 
                mem_attn_mask
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
        self._time_axis = 1 if self.layout == 'NT' else 0

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

    def forward(self, hidden_states, valid_length): 
        # 1. relative position embeddings and attention masks
        position_embeddings = self.relative_position_encoder(
            gen_rel_position(hidden_states, layout=self.layout)
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
        self._time_axis = 1 if self.layout == 'NT' else 0

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

    def _init_key_values(self, batch_size, ctx, dtype='float32'): 
        return list(layer._init_key_value(batch_size, ctx, dtype) for layer in self.layers)

    def incremental_decode(
        self, 
        step_hidden_states, 
        position, 
        past_key_values, 
        mem_states, 
        mem_valid_length
    ): 
        # 1. relative position embeddings and attention mask
        # step_position_embeddings: Shape (num_heads, 1, L_seq), for self-attention
        # step_mem_attn_mask: Shape (B, 1, L_mem), for cross-attention
        position_embeddings = self.relative_position_encoder(
            gen_rel_position(
                step_hidden_states, 
                past_data=past_key_values[0][0], 
                layout=self.layout
            )
        )
        step_position_embeddings = position_embeddings[:, -1:, :]
        step_mem_attn_mask = gen_mem_attn_mask(
            mem_states, 
            mem_valid_length, 
            step_hidden_states, 
            dtype=self._dtype, 
            layout=self.layout
        )

        # 2. decoder blocks and other layers
        step_hidden_states = self.dropout(step_hidden_states)
        present_key_values = []
        for i, layer in enumerate(self.layers): 
            step_hidden_states, present_key_value = layer.incremental_decode(
                step_hidden_states, 
                step_position_embeddings, 
                past_key_values[i], 
                mem_states, 
                step_mem_attn_mask
            )
            present_key_values.append(present_key_value)
        step_hidden_states = self.final_layer_norm(step_hidden_states)
        step_hidden_states = self.dropout(step_hidden_states)
        return step_hidden_states, present_key_values

    def forward(self, hidden_states, valid_length, mem_states, mem_valid_length): 
        # 1. relative position embeddings and attention masks
        # position_embeddings: Shape (num_heads, L_seq, L_seq), broadcastable, for self-attention 
        # self_attn_mask: Shape (B, L_seq, L_seq), for self-attention
        # mem_attn_mask: Shape (B, L_seq, L_mem), for cross-attention
        position_embeddings = self.relative_position_encoder(
            gen_rel_position(hidden_states, layout=self.layout)
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
                mem_attn_mask
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
        self._d_model = d_model
        self._d_kv = d_kv
        self._d_ff = d_ff
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._inner_dim = num_heads * d_kv
        self._activation = activation
        self._init_factor = init_factor
        self._dtype = dtype
        assert layout in ['TN', 'NT'], \
            'Invalid layout: {}. Only "TN" and "NT" are supported.'.format(layout)
        self._layout = layout
        self._time_axis = 1 if self.layout == 'NT' else 0

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

    def encode(self, src_data, src_valid_length): 
        src_hidden_states = self.input_embedding_layer(src_data)
        enc_out = self.encoder(
            src_hidden_states, 
            src_valid_length
        )
        return enc_out

    def decode(self, tgt_data, tgt_valid_length, mem_states, mem_valid_length): 
        tgt_hidden_states = self.input_embedding_layer(tgt_data)
        dec_out = self.decoder(
            tgt_hidden_states, 
            tgt_valid_length, 
            mem_states, 
            mem_valid_length
        )
        return dec_out

    def forward(self, src_data, src_valid_length, tgt_data, tgt_valid_length): 
        enc_out = self.encode(src_data, src_valid_length)
        dec_out = self.decode(tgt_data, tgt_valid_length, enc_out, src_valid_length)
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


def mask_to_sentinel(tokens, noise_mask, vocab_size): 
    if isinstance(tokens, list) and isinstance(tokens[0], list): 
        masked_tokens = []
        for i, (tok, mask) in enumerate(zip(tokens, noise_mask)): 
            masked_tokens.append(mask_to_sentinel(tok, mask, vocab_size))
        return masked_tokens
    elif isinstance(tokens, list): 
        # reference: https://github.com/google-research/text-to-text-transfer-transformer/blob/867715664c8393cf12093ea9633f868c0df35548/t5/data/preprocessors.py#L2802-L2839
        assert isinstance(tokens, list) and isinstance(noise_mask, list), 'Only Python lists are supported'
        assert len(tokens) == len(noise_mask), 'tokens and noise_mask have different shapes'
        # converting back to numpy array is an ad hoc solution to bugs in mxnet.np.pad()
        tokens = _np.array(tokens)
        noise_mask = _np.array(noise_mask)
        prev_token_is_noise = _np.pad(noise_mask[:-1], [[1, 0]])
        first_noise_tokens = _np.logical_and(noise_mask, _np.logical_not(prev_token_is_noise))
        subsequent_noise_tokens = _np.logical_and(noise_mask, prev_token_is_noise)
        sentinel = vocab_size - _np.cumsum(first_noise_tokens.astype(tokens.dtype))
        tokens = _np.where(first_noise_tokens, sentinel, tokens)
        return tokens[_np.logical_not(subsequent_noise_tokens)].tolist()
    else: 
        raise ValueError('Unsupported input type: {}'.format(tokens))


@use_np
class T5Inference(HybridBlock, BaseStepDecoder): 
    def __init__(self, model): 
        super().__init__()
        self.model = model
        self.output_layer = nn.Dense(
            units=model.vocab_size, 
            in_units=model._d_model, 
            flatten=False, 
            use_bias=False, 
            dtype=model._dtype
        )
        self.output_layer.weight = model.input_embedding_layer.weight

    def initialize(self, **kwargs): 
        raise NotImplementedError(
            'You can not initialize a T5Inference Model! ' \
            'The correct approach is to create a T5Model and ' \
            'then feed it into a T5Inference.'
        )

    @property
    def state_batch_axis(self): 
        """The returned 4-tuple corresponds to the batch axes of
        results of `init_states()`

        Returns
        -------
        enc_out_batch_axis
        src_valid_length_batch_axis
        position_batch_axis
        dec_layer_batch_axis
        """
        if self.model.layout == 'NT':
            return 0, 0, 0, self.model.decoder.state_batch_axis
        else:
            return 1, 0, 0, self.model.decoder.state_batch_axis

    def init_states(self, src_data, src_valid_length): 
        batch_size = src_data.shape[1 - self.model._time_axis] # NT: 0; TN: 1
        ctx = src_data.ctx
        enc_out = self.model.encode(src_data, src_valid_length)
        position = np.zeros((batch_size,), dtype=np.int32, ctx=ctx)
        key_values = self.model.decoder._init_key_values(batch_size, ctx, dtype=enc_out.dtype)
        return enc_out, src_valid_length, position, key_values

    def forward(self, step_data, past_states): 
        mem_states, mem_valid_length, position, past_key_values = past_states
        step_hidden_states = self.model.input_embedding_layer(step_data)
        # NT: (B, d_model) -> (B, 1, d_model); TN: (B, d_model) -> (1, B, d_model)
        step_hidden_states = np.expand_dims(step_hidden_states, axis=self.model._time_axis)
        step_hidden_states, present_key_values = self.model.decoder.incremental_decode(
            step_hidden_states, 
            position, 
            past_key_values, 
            mem_states, 
            mem_valid_length
        )
        step_hidden_states = self.output_layer(step_hidden_states)
        # NT: (B, 1, vocab_size) -> (B, vocab_size); TN: (1, B, vocab_size) -> (B, vocab_size)
        step_hidden_states = npx.reshape(step_hidden_states, (-5, -1))
        return step_hidden_states, (mem_states, mem_valid_length, position + 1, present_key_values)


class T5NMTInference(T5Inference): 
    def __init__(self, *args, **kwargs): 
        print(
            'Note: T5NMTInference is deprecated. We have renamed it to T5Inference and ' \
            'migrated all previous functionalities. Please use it instead.'
        )
        super().__init__(*args, **kwargs)


class T5Tokenizer(SentencepieceTokenizer): 
    """This inheriting class is capable of handling extra tokens which do not present in self._sp_model
    """
    def __init__(self, vocab_path, extra_ids=100): 
        # extend tokens in vocab with <extra_id>s, which correspond to noise span sentinels one-by-one
        # <extra_id_0> will the last token in the new vocabulary
        special_tokens = {
            'extra{}_token'.format(i): '<extra_id_{}>'.format(i) for i in range(extra_ids - 1, -1, -1)
        }
        spiece_model = SentencepieceTokenizer(vocab_path)
        tokens = spiece_model.vocab.all_tokens
        tokens.extend(list(special_tokens.values()))
        # re-specify special tokens 
        special_tokens['eos_token'] = spiece_model.vocab.eos_token
        special_tokens['unk_token'] = spiece_model.vocab.unk_token
        special_tokens['pad_token'] = spiece_model.vocab.pad_token
        super().__init__(
            model_path=vocab_path, 
            vocab=Vocab(tokens, **special_tokens), 
            lowercase=False
        )

    def _filter_extra_tokens(self, tokens): 
        def _filter(tokens, token_type): 
            if token_type is str: 
                return [token for token in tokens if 'extra_id' not in token]
            elif token_type is int: 
                return [token for token in tokens if token < len(self._sp_model)]
        is_multi_sentences = is_tokens_from_multiple_sentences(tokens)
        token_type = get_token_type(tokens)
        if not is_multi_sentences: 
            return _filter(tokens, token_type)
        else: 
            return [_filter(ele_token, token_type) for ele_token in tokens]

    def decode(self, tokens): 
        tokens = self._filter_extra_tokens(tokens)
        return super().decode(tokens)


def list_pretrained_t5(): 
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_t5(
    model_name: str = 'google_t5_base', 
    root: str = get_model_zoo_home_dir(), 
    load_backbone: bool = True, 
    extra_ids: int = 100
) -> Tuple[CN, T5Tokenizer, str, str]: 
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
    # lm model simply uses T5Model as backbone, so no additional params
    local_lm_params_path = None
    tokenizer = T5Tokenizer(local_paths['vocab'], extra_ids)
    if cfg is None: 
        cfg = T5Model.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_lm_params_path


BACKBONE_REGISTRY.register(
    't5', 
    [T5Model, get_pretrained_t5, list_pretrained_t5]
)
