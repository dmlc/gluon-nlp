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
GPT-2 Model

@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
"""

__all__ = ['GPT2Model', 'GPT2ForLM', 'list_pretrained_gpt2', 'get_pretrained_gpt2']

import os
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn
from ..registry import BACKBONE_REGISTRY
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..utils.registry import Registry
from ..initializer import TruncNorm
from ..attention_cell import MultiHeadAttentionCell
from ..layers import get_activation, PositionalEmbedding
from ..data.tokenizers import HuggingFaceByteBPETokenizer


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'gpt2.txt'))
gpt2_cfg_reg = Registry('gpt2_cfg')


@gpt2_cfg_reg.register()
def gpt2_124M():
    cfg = CN()
    # Config for the roberta base model
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 50257
    cfg.MODEL.units = 768
    cfg.MODEL.max_length = 1024
    cfg.MODEL.num_heads = 12
    cfg.MODEL.num_layers = 12
    cfg.MODEL.embed_dropout_prob = 0.1
    cfg.MODEL.hidden_dropout_prob = 0.1
    cfg.MODEL.attention_dropout_prob = 0.1
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.activation = 'gelu(tanh)'
    cfg.MODEL.layer_norm_eps = 1E-5
    cfg.MODEL.dtype = 'float32'
    # Layout
    cfg.MODEL.layout = 'NT'
    cfg.MODEL.compute_layout = 'auto'
    # Initialization method
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg

@gpt2_cfg_reg.register()
def gpt2_355M():
    cfg = gpt2_124M()
    cfg.defrost()
    cfg.MODEL.num_heads = 16
    cfg.MODEL.num_layers = 24
    cfg.MODEL.units = 1024
    cfg.freeze()
    return cfg

@gpt2_cfg_reg.register()
def gpt2_774M():
    cfg = gpt2_124M()
    cfg.defrost()
    cfg.MODEL.num_heads = 20
    cfg.MODEL.num_layers = 36
    cfg.MODEL.units = 1280
    cfg.freeze()
    return cfg

PRETRAINED_URL = {
    'gpt2_124M': {
        'cfg': gpt2_124M(),
        'merges': 'gpt2_124M/gpt2-396d4d8e.merges',
        'vocab': 'gpt2_124M/gpt2-9dc62091.vocab',
        'params': 'gpt2_124M/model-bfed311d.params',
        'lm_params': 'gpt2_124M/model_lm-99b90604.params'
    },
    'gpt2_355M': {
        'cfg': gpt2_355M(),
        'merges': 'gpt2_355M/gpt2-396d4d8e.merges',
        'vocab': 'gpt2_355M/gpt2-9dc62091.vocab',
        'params': 'gpt2_355M/model-81dee612.params',
        'lm_params': 'gpt2_355M/model_lm-eed0e964.params'
    },
    'gpt2_774M': {
        'cfg': gpt2_774M(),
        'merges': 'gpt2_774M/gpt2-396d4d8e.merges',
        'vocab': 'gpt2_774M/gpt2-9dc62091.vocab',
        'params': 'gpt2_774M/model-9917e24e.params',
        'lm_params': 'gpt2_774M/model_lm-cfbfa641.params'
    },
}


@use_np
class GPT2SelfAttentionLayer(HybridBlock):
    def __init__(self, units: int = 768,
                 num_heads: int = 12,
                 layer_norm_eps: float = 1E-5,
                 use_qkv_bias: bool = True,
                 hidden_dropout_prob: float = 0.1,
                 attention_dropout_prob: float = 0.1,
                 weight_initializer=None,
                 bias_initializer='zeros',
                 dtype='float32',
                 layout='NT'):
        super().__init__()
        self._units = units
        self._num_heads = num_heads
        self._layer_norm_eps = layer_norm_eps
        self._use_qkv_bias = use_qkv_bias
        self._hidden_dropout_prob = hidden_dropout_prob
        self._attention_dropout_prob = attention_dropout_prob
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._dtype = dtype
        self._layout = layout
        assert layout in ['TN', 'NT'], 'Invalid layout received = {}. ' \
                                       'Only "TN" and "NT" are accepted!'.format(layout)
        self._attention_layout = 'NTK' if self._layout == 'NT' else 'TNK'
        
        self.ln = nn.LayerNorm(
            epsilon=self._layer_norm_eps,
            in_channels=self._units
        )
        self.qkv = nn.Dense(
            3 * units,
            in_units=units,
            use_bias=use_qkv_bias,
            flatten=False,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            dtype=dtype
        )
        self.out_proj = nn.Dense(
            units=units,
            in_units=units,
            use_bias=True,
            flatten=False,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            dtype=self._dtype
        )
        self.attention_cell = MultiHeadAttentionCell(
            query_units=self._units,
            num_heads=self._num_heads,
            attention_dropout=self._attention_dropout_prob,
            scaled=True,
            dtype=self._dtype,
            layout=self._attention_layout
        )
        self.hidden_dropout = nn.Dropout(self._hidden_dropout_prob)

    def hybrid_forward(self, F, x, layer_states, prev_len):
        """

        Parameters
        ----------
        x :
            - layout = 'NT'
                Shape (batch_size, seq_length, C_in)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_in)
        layer_states :
            - layout = 'NT'
                Shape (2, batch_size, prev_len, C_in)
            - layout = 'TN'
                Shape (2, prev_len, batch_size, C_in)
        prev_len
        """
        x = self.ln(x)
        if self._layout == 'NT':
            batch_axis, time_axis = 0, 1
        else:
            batch_axis, time_axis = 1, 0

        query, key, value = F.np.split(self.qkv(x), 3, axis=-1)
        if layer_states is not None:
            prev_key, prev_value = layer_states[0], layer_states[1]
            key = F.np.concatenate([prev_key, key], axis=time_axis)
            value = F.np.concatenate([prev_value, value], axis=time_axis)
        new_states = F.np.stack([key, value], axis=0)
        
        # gen mask
        query_pos = F.npx.arange_like(query, axis=time_axis)
        if prev_len is not None:
            query_pos = query_pos + prev_len
        key_pos = F.npx.arange_like(key, axis=time_axis)
        # (query_len, key_len)
        mask = (F.npx.reshape(key_pos, (1, -1)) <= 
                F.npx.reshape(query_pos, (-1, 1))).astype(self._dtype)
        # broadcast to (batch_size, query_len, key_len)
        mask = F.npx.broadcast_like(
            F.np.expand_dims(mask, axis=0),
            query,
            lhs_axes=0,
            rhs_axes=batch_axis
        )

        query = F.npx.reshape(query, (-2, -2, self._num_heads, -1))
        key = F.npx.reshape(key, (-2, -2, self._num_heads, -1))
        value = F.npx.reshape(value, (-2, -2, self._num_heads, -1))

        out, [_, attn_weight] = self.attention_cell(query, key, value, mask)
        out = self.out_proj(out)
        out = self.hidden_dropout(out)

        return out, new_states


@use_np
class GPT2FFN(HybridBlock):
    def __init__(self,
                 units: int = 768,
                 hidden_size: int = 3072,
                 layer_norm_eps: float = 1E-5,
                 hidden_dropout_prob: float = 0.1,
                 weight_initializer=None,
                 bias_initializer='zeros',
                 activation='gelu(tanh)',
                 dtype='float32'):
        super().__init__()
        self._units = units
        self._hidden_size = hidden_size
        self._layer_norm_eps = layer_norm_eps
        self._hidden_dropout_prob = hidden_dropout_prob
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._activation = activation
        self._dtype = dtype
        self.layer_norm = nn.LayerNorm(epsilon=self._layer_norm_eps,
                                       in_channels=self._units)
        self.ffn_1 = nn.Dense(units=self._hidden_size,
                              in_units=self._units,
                              flatten=False,
                              weight_initializer=self._weight_initializer,
                              bias_initializer=self._bias_initializer,
                              dtype=self._dtype)
        self.activation = get_activation(self._activation)
        self.ffn_2 = nn.Dense(units=self._units,
                              in_units=self._hidden_size,
                              flatten=False,
                              weight_initializer=self._weight_initializer,
                              bias_initializer=self._bias_initializer,
                              dtype=self._dtype)
        self.hidden_dropout = nn.Dropout(self._hidden_dropout_prob)

    def hybrid_forward(self, F, data):
        # here the residual connection is applied before the layernorm,
        # which is different from the PositionwiseFFN(pre_norm=True)
        out = self.layer_norm(data)
        out = self.activation(self.ffn_1(out))
        out = self.ffn_2(out)
        out = self.hidden_dropout(out)
        out = out + data
        return out


@use_np
class GPT2Layer(HybridBlock):
    def __init__(self, units: int = 768,
                 num_heads: int = 12,
                 layer_norm_eps: float = 1E-5,
                 use_qkv_bias: bool = True,
                 hidden_dropout_prob: float = 0.1,
                 attention_dropout_prob: float = 0.1,
                 weight_initializer=None,
                 bias_initializer='zeros',
                 activation: str = 'gelu(tanh)',
                 dtype='float32',
                 layout='NT'):
        super().__init__()
        self._units = units
        self._hidden_size = 4 * units
        self._num_heads = num_heads
        self._layer_norm_eps = layer_norm_eps
        self._use_qkv_bias = use_qkv_bias
        self._hidden_dropout_prob = hidden_dropout_prob
        self._attention_dropout_prob = attention_dropout_prob
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._activation = activation
        self._dtype = dtype
        self._layout = layout
        
        self.atten = GPT2SelfAttentionLayer(
            units=self._units,
            num_heads=self._num_heads,
            layer_norm_eps=self._layer_norm_eps,
            use_qkv_bias=self._use_qkv_bias,
            weight_initializer=self._weight_initializer,
            bias_initializer=self._bias_initializer,
            dtype=self._dtype,
            layout=self._layout
        )
        self.ffn = GPT2FFN(
            units=self._units,
            hidden_size=self._hidden_size,
            layer_norm_eps=self._layer_norm_eps,
            hidden_dropout_prob=self._hidden_dropout_prob,
            weight_initializer=self._weight_initializer,
            bias_initializer=self._bias_initializer,
            activation=self._activation,
            dtype=self._dtype
        )
    
    def hybrid_forward(self, F, x, layer_states, prev_len):
        """

        Parameters
        ----------
        x
            Input
            - layout = 'NT'
                Shape (batch_size, seq_length, C_in)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_in)
        layer_states :
            - layout = 'NT'
                Shape (2, batch_size, prev_len, C_in)
            - layout = 'TN'
                Shape (2, prev_len, batch_size, C_in)
        prev_len
            The previous length

        Returns
        -------
        new_x
            Output
            - layout = 'NT'
                Shape (batch_size, seq_length, C_out)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_out)
        new_states
            - layout = 'NT'
                Shape (2, batch_size, prev_len + seq_length, C_in)
            - layout = 'TN'
                Shape (2, prev_len + seq_length, batch_size, C_in)
        """
        h, new_layer_states = self.atten(x, layer_states, prev_len)
        x = x + h
        h = self.ffn(x)
        return h, new_layer_states


@use_np
class GPT2Model(HybridBlock):
    def __init__(self,
                 vocab_size=50257,
                 units=768,
                 num_layers=12,
                 num_heads=12,
                 max_length=1024,
                 pos_embed_type='learned',
                 activation='gelu(tanh)',
                 layer_norm_eps=1E-5,
                 embed_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 embed_initializer=TruncNorm(stdev=0.02),
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 dtype='float32',
                 output_all_encodings=False,
                 layout='NT',
                 compute_layout='auto'):
        super().__init__()
        self._vocab_size = vocab_size
        self._units = units
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._max_length = max_length
        self._pos_embed_type = pos_embed_type
        self._activation = activation
        self._layer_norm_eps = layer_norm_eps
        self._embed_dropout_prob = embed_dropout_prob
        self._hidden_dropout_prob = hidden_dropout_prob
        self._attention_dropout_prob = attention_dropout_prob
        self._embed_initializer = embed_initializer
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._dtype = dtype
        self._layout = layout
        if compute_layout == 'auto' or compute_layout is None:
            self._compute_layout = layout
        else:
            self._compute_layout = compute_layout
        self._embed = nn.Embedding(
            input_dim=self._vocab_size,
            output_dim=self._units,
            weight_initializer=embed_initializer,
            dtype=self._dtype
        )
        self._embed_dropout = nn.Dropout(self._embed_dropout_prob)
        self._pos_embed = PositionalEmbedding(
            units=self._units,
            max_length=self._max_length,
            dtype=self._dtype,
            method=pos_embed_type
        )
        self._layers = nn.HybridSequential()
        for layer_idx in range(self._num_layers):
            self._layers.add(
                GPT2Layer(
                    units=self._units,
                    num_heads=self._num_heads,
                    layer_norm_eps=self._layer_norm_eps,
                    use_qkv_bias=True,
                    hidden_dropout_prob=self._hidden_dropout_prob,
                    attention_dropout_prob=self._attention_dropout_prob,
                    weight_initializer=self._weight_initializer,
                    bias_initializer=self._bias_initializer,
                    activation=self._activation,
                    dtype=self._dtype,
                    layout=self._compute_layout
                )
            )
        self._final_ln = nn.LayerNorm(epsilon=layer_norm_eps,
                                      in_channels=units)

    @property
    def layout(self):
        return self._layout

    def hybrid_forward(self, F, x, states, prev_len):
        """

        Parameters
        ----------
        x
            Input
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        states
            The previous states
            - layout = 'NT'
                Shape (num_layers, 2, batch_size, prev_len, C_in)]
            - layout = 'TN'
                Shape (num_layers, 2, prev_len, batch_size, C_in)]
        prev_len
            The previous length. It will be a scalar.

        Returns
        -------
        new_x
            Output
            - layout = 'NT'
                Shape (batch_size, seq_length, C_out)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_out)
        new_states
            The new states
            - layout = 'NT'
                Shape (num_layers, 2, batch_size, prev_len + seq_length, C_in)
            - layout = 'TN'
                Shape (num_layers, 2, prev_len + seq_length, batch_size, C_in)
        """
        x = self.get_initial_embedding(F, x, prev_len)
        
        if self._layout != self._compute_layout:
            x = F.np.swapaxes(x, 0, 1)
            states = F.np.swapaxes(states, 2, 3)
        
        new_states = []
        for layer_idx in range(self._num_layers):
            layer_states = None if states is None else states[layer_idx]
            x, new_layer_states = self._layers[layer_idx](x, layer_states, prev_len)
            new_states.append(new_layer_states)
        new_states = F.np.stack(new_states, axis=0)
        
        x = self._final_ln(x)
        if self._layout != self._compute_layout:
            x = F.np.swapaxes(x, 0, 1)
            new_states = F.np.swapaxes(new_states, 2, 3)
        return x, new_states

    def get_initial_embedding(self, F, inputs, prev_len):
        """Get the initial token embeddings that considers the token type and positional embeddings

        Parameters
        ----------
        F
        inputs
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        prev_len
            The previous length. It will be a scalar.
        
        Returns
        -------
        embedding
            - layout = 'NT'
                Shape (batch_size, seq_length, C)
            - layout = 'TN'
                Shape (seq_length, batch_size, C)
        """
        embedding = self._embed(inputs)
        if self._layout == 'NT':
            batch_axis, time_axis = 0, 1
        else:
            batch_axis, time_axis = 1, 0
        if self._pos_embed_type is not None:
            pos = F.npx.arange_like(inputs, axis=time_axis)
            if prev_len is not None:
                pos = pos + prev_len
            positional_embedding = self._pos_embed(pos)
            positional_embedding = F.np.expand_dims(positional_embedding, axis=batch_axis)
            embedding = embedding + positional_embedding
        embedding = self._embed_dropout(embedding)
        return embedding

    def init_states(self, batch_size, ctx):
        """Initialize the states required for incremental decoding

        Returns
        -------
        init_states
            - layout = 'NT'
                Shape (num_layers, 2, batch_size, 0, C_in)
            - layout = 'TN'
                Shape (num_layers, 2, 0, batch_size, C_in)
        """
        return mx.np.zeros(shape=(self._num_layers, 2, batch_size, 0,
                                  self._units), ctx=ctx, dtype=self._dtype) if self.layout == 'NT' else \
               mx.np.zeros(shape=(self._num_layers, 2, 0, batch_size,
                                  self._units), ctx=ctx, dtype=self._dtype)

    @staticmethod
    def get_cfg(key=None):
        if key is not None:
            return gpt2_cfg_reg.create(key)
        else:
            return gpt2_124M()

    @classmethod
    def from_cfg(cls,
                 cfg,
                 dtype=None,
                 output_all_encodings=False) -> 'GPT2Model':
        cfg = GPT2Model.get_cfg().clone_merge(cfg)
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        if dtype is None:
            dtype = cfg.MODEL.dtype
        return cls(vocab_size=cfg.MODEL.vocab_size,
                   units=cfg.MODEL.units,
                   num_layers=cfg.MODEL.num_layers,
                   num_heads=cfg.MODEL.num_heads,
                   max_length=cfg.MODEL.max_length,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   activation=cfg.MODEL.activation,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps,
                   embed_dropout_prob=cfg.MODEL.embed_dropout_prob,
                   hidden_dropout_prob=cfg.MODEL.hidden_dropout_prob,
                   attention_dropout_prob=cfg.MODEL.attention_dropout_prob,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   dtype=dtype,
                   output_all_encodings=output_all_encodings,
                   layout=cfg.MODEL.layout,
                   compute_layout=cfg.MODEL.compute_layout)


@use_np
class GPT2ForLM(HybridBlock):
    def __init__(self, backbone_cfg=None):
        super().__init__()
        self._backbone_model = GPT2Model.from_cfg(backbone_cfg)
        self._lm_head = nn.Dense(
            units=backbone_cfg.MODEL.vocab_size,
            in_units=backbone_cfg.MODEL.units,
            use_bias=False,
            flatten=False
        )
        self._lm_head.weight = self._backbone_model._embed.weight

    def hybrid_forward(self, F, inputs, states, prev_len):
        """Getting the logits

        Parameters
        ----------
        F
        inputs
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        states
            The states.
            - layout = 'NT'
                Shape (num_layers, 2, batch_size, prev_len, C_in)
            - layout = 'TN'
                Shape (num_layers, 2, prev_len, batch_size, C_in)
        prev_len
            Will be a scalar that represents the previous length

        Returns
        -------
        logits
            - layout = 'NT'
                Shape (batch_size, seq_length, vocab_size).
            - layout = 'TN'
                Shape (seq_length, batch_size, vocab_size).
        new_states
            - layout = 'NT'
                Shape (num_layers, 2, batch_size, prev_len + seq_length, C_in)
            - layout = 'TN'
                Shape (num_layers, 2, prev_len + seq_length, batch_size, C_in)
        """
        contextual_embeddings, new_states = self._backbone_model(inputs, states, prev_len)
        logits = self._lm_head(contextual_embeddings)
        return logits, new_states

    def init_states(self, batch_size, ctx):
        return self._backbone_model.init_states(batch_size, ctx)


def list_pretrained_gpt2():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_gpt2(model_name: str = 'gpt2_124M',
                        root: str = get_model_zoo_home_dir(),
                        load_backbone: bool = True,
                        load_lm: bool = False)\
        -> Tuple[CN, HuggingFaceByteBPETokenizer, str, str]:
    """Get the pretrained GPT-2 weights

    Parameters
    ----------
    model_name
        The name of the GPT-2 model.
    root
        The downloading root
    load_backbone
        Whether to load the weights of the backbone network
    load_lm
        Whether to load the weights of LM

    Returns
    -------
    cfg
        Network configuration
    tokenizer
        The HuggingFaceByteBPETokenizer
    params_path
        Path to the parameters
    lm_params_path
        Path to the parameter that includes both the backbone and the LM
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_gpt2())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None
    merges_path = PRETRAINED_URL[model_name]['merges']
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']
    lm_params_path = PRETRAINED_URL[model_name]['lm_params']

    local_paths = dict()
    download_jobs = [('vocab', vocab_path), ('merges', merges_path)]
    if cfg is None:
        download_jobs.append(('cfg', cfg_path))
    for k, path in download_jobs:
        local_paths[k] = download(url=get_repo_model_zoo_url() + path,
                                  path=os.path.join(root, path),
                                  sha1_hash=FILE_STATS[path])
    if load_backbone:
        local_params_path = download(url=get_repo_model_zoo_url() + params_path,
                                     path=os.path.join(root, params_path),
                                     sha1_hash=FILE_STATS[params_path])
    else:
        local_params_path = None
    if load_lm and lm_params_path is not None:
        local_lm_params_path = download(url=get_repo_model_zoo_url() + lm_params_path,
                                        path=os.path.join(root, lm_params_path),
                                        sha1_hash=FILE_STATS[lm_params_path])
    else:
        local_lm_params_path = None

    tokenizer = HuggingFaceByteBPETokenizer(
                    merges_file=local_paths['merges'],
                    vocab_file=local_paths['vocab'])
    if cfg is None:
        cfg = GPT2Model.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_lm_params_path


BACKBONE_REGISTRY.register('gpt2', [GPT2Model,
                                    get_pretrained_gpt2,
                                    list_pretrained_gpt2])
