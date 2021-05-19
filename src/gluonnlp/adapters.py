import math
from collections import OrderedDict
import mxnet as mx
from mxnet import np, npx
from mxnet import use_np
from mxnet.gluon import nn, HybridBlock, Parameter, Constant
import numpy as _np
from .attention_cell import MultiHeadAttentionCell
import mxnet.numpy_extension as _mx_npx
from .layers import get_activation, get_norm_layer, _gen_repr_with_kwargs


__all__ = ['BasicAdapter', 'AdapterFusion', 'AdapterModule', 'PositionwiseFFN_adapter']

@use_np
class BasicAdapter(nn.HybridBlock):
    def __init__(self, config: dict, in_units: int):
        super().__init__()
        self._units = config['units']
        self._activation = config['activation']
        self.down_proj = nn.Dense(in_units=in_units,
                                  units=self._units,
                                  flatten=False,
                                  weight_initializer=None, bias_initializer='zero')
        self.activate = get_activation(self._activation)
        self.up_proj = nn.Dense(in_units=self._units,
                                  units=in_units,
                                  flatten=False,
                                  weight_initializer=None, bias_initializer='zero')



    def forward(self, data, residual):
        out = self.down_proj(data)
        out = self.activate(out)
        out = self.up_proj(out)
        return out + residual


@use_np
class AdapterFusion(nn.HybridBlock):
    def __init__(self, in_units):
        super().__init__()
        self.query_proj = nn.Dense(in_units=in_units, units=in_units, flatten=False, weight_initializer=None, bias_initializer='zero')
        self.key_proj = nn.Dense(in_units=in_units, units=in_units, flatten=False, weight_initializer=None, bias_initializer='zero')
        self.value_proj = nn.Dense(in_units=in_units, units=in_units, flatten=False, weight_initializer=None, bias_initializer='zero')
        '''
        self.attention_cell = MultiHeadAttentionCell(query_units=in_units,
                                                     num_heads=1,
                                                     attention_dropout=0,
                                                     scaled=True)
        '''

    def forward(self, query, key, value):
        #query bs, length, unit
        #key bs, length, num_adapters, unit


        key = self.key_proj(key).transpose((0, 1, 3, 2))
        value = self.value_proj(value)
        # query = npx.reshape(self.query_proj(query), (-2, -2, 1, -1))
        query = self.query_proj(query)
        #scores = np.squeeze(npx.batch_dot(query, key), axis=2)
        scores = np.einsum('blu, blun -> bln', query, key)
        attn_weights = npx.softmax(scores, axis=-1)
        #attn batch size lenght, num
        #value bs l, num, u
        output = np.einsum('bln, blnu -> blu', attn_weights, value)
        #output = np.squeeze(npx.batch_dot(npx.reshape(attn_weights, (-2, -2, 1, -1)),  value), axis=2)
        return output

@use_np
def get_base_adapter(base_adapter_config, in_units):
    if base_adapter_config['type'] == 'Basic':
        base_adapter =  BasicAdapter(config=base_adapter_config, in_units=in_units)
    else:
        pass
    return base_adapter

@use_np
class AdapterModule(nn.HybridBlock):
    def __init__(self, in_units:int, adapter_config:dict):
        super().__init__()
        self._in_units = in_units
        self._adapter_config = adapter_config
        self._basic_num = 0
        self.base_adapter_stacks = nn.HybridSequential()
        for name in adapter_config['task_names']:
            self.base_adapter_stacks.add(get_base_adapter(adapter_config[name], in_units))
            self._basic_num += 1
        if adapter_config['adapter_fusion']:
            self.adapter_fusion = AdapterFusion(in_units)
        if adapter_config['pre_operator']:
            self.pre_norm = nn.LayerNorm(epsilon=adapter_config['layer_norm_eps'],
                                       in_channels=in_units)


    def forward(self, data, residual):
        new_residual = data
        if self._adapter_config['pre_operator']:
            data = data + residual
            data = self.pre_norm(data)

        output = []
        for layer_idx in range(self._basic_num):
            layer = self.base_adapter_stacks[layer_idx]
            output.append(layer(data, new_residual))

        if  self._adapter_config['adapter_fusion']:
            output = np.stack(output, axis=2)
            #output = np.concatenate(output, axis = 1)
            output = self.adapter_fusion(new_residual, output, output)

            return output
        else:
            return output[0]



@use_np
class PositionwiseFFN_adapter(HybridBlock):
    """The Position-wise FFN layer used in Transformer-like architectures,
    # this architecture copy from layers.py to aviod import probles

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    """
    def __init__(self,
                 units: int = 512,
                 hidden_size: int = 2048,
                 use_bias=True,
                 activation_dropout: float = 0.0,
                 dropout: float = 0.1,
                 weight_initializer=None,
                 bias_initializer='zeros',
                 activation='relu',
                 use_gated_activation=False,
                 normalization: str = 'layer_norm',
                 layer_norm_eps: float = 1E-5,
                 pre_norm: bool = False,
                 dtype='float32',
                 use_adapter='False',
                 adapter_config={},
                 **kwargs):
        """

        Parameters
        ----------
        units
        hidden_size
        activation_dropout
        dropout
        weight_initializer
        bias_initializer
        activation
        normalization
            layer_norm or no_norm
        layer_norm_eps
        pre_norm
            Pre-layer normalization as proposed in the paper:
            "[ACL2018] The Best of Both Worlds: Combining Recent Advances in
             Neural Machine Translation"
            This will stabilize the training of Transformers.
            You may also refer to
            "[Arxiv2020] Understanding the Difficulty of Training Transformers"
        """
        super().__init__()
        self._dtype = dtype
        self._pre_norm = pre_norm
        self._use_gated_activation = use_gated_activation
        self._use_adapter = use_adapter
        self._adapter_config = adapter_config
        self._kwargs = OrderedDict([
            ('units', units),
            ('hidden_size', hidden_size),
            ('activation_dropout', activation_dropout),
            ('activation', activation),
            ('dropout', dropout),
            ('normalization', normalization),
            ('layer_norm_eps', layer_norm_eps),
            ('pre_norm', pre_norm),
            ('dtype', self._dtype)
        ])
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_dropout_layer = nn.Dropout(activation_dropout)
        self.ffn_1 = nn.Dense(units=hidden_size,
                              in_units=units,
                              flatten=False,
                              use_bias=use_bias,
                              weight_initializer=weight_initializer,
                              bias_initializer=bias_initializer,
                              dtype=dtype)
        if use_gated_activation:
            self.gated_ffn_1 = nn.Dense(units=hidden_size,
                                        in_units=units,
                                        flatten=False,
                                        use_bias=use_bias,
                                        weight_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        dtype=dtype)
        self.activation = get_activation(activation)
        self.ffn_2 = nn.Dense(units=units,
                              in_units=hidden_size,
                              flatten=False,
                              use_bias=use_bias,
                              weight_initializer=weight_initializer,
                              bias_initializer=bias_initializer,
                              dtype=dtype)
        # TODO(sxjscience) We may need to set the dtype flag in LayerNorm, need to double check
        self.layer_norm = get_norm_layer(in_channels=units,
                                         normalization=normalization,
                                         epsilon=layer_norm_eps,
                                         **kwargs)
        if self._use_adapter and 'location_1' in self._adapter_config:
            self.adapter_layer_ffn = AdapterModule(in_units=units, adapter_config=adapter_config['location_1'])

    def forward(self, data):
        """

        Parameters
        ----------
        F
        data :
            Shape (B, seq_length, C_in)

        Returns
        -------
        out :
            Shape (B, seq_length, C_out)
        """
        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        if self._use_gated_activation:
            gated_out = self.activation(self.gated_ffn_1(data))
            out = gated_out * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        if self._use_adapter and 'location_1' in self._adapter_config:
            out = self.adapter_layer_ffn(out, residual)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out

    def __repr__(self):
        return _gen_repr_with_kwargs(self._kwargs, self.__class__.__name__)