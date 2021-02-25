__all__ = ['TransformerEncoderLayer', 'TransformerDecoderLayer', 'TransformerDecoder']

from typing import Union, Optional, List
from torch import nn
import torch as th
from ..layers import PositionwiseFFN
from ..attention_cell import MultiHeadAttentionCell, gen_mem_attn_mask, gen_self_attn_mask
from ..utils import to_torch_dtype


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, units: int = 512, hidden_size: int = 2048, num_heads: int = 8,
                 attention_dropout_prob: float = 0.1, hidden_dropout_prob: float = 0.1,
                 activation_dropout_prob: float = 0.0, layer_norm_eps: float = 1e-12,
                 pre_norm: bool = False, use_qkv_bias: bool = True, activation: str = 'relu',
                 layout='NT'):
        """
        Parameters
        ----------
        units
        hidden_size
        num_heads
        attention_dropout_prob
        hidden_dropout_prob
        activation_dropout_prob
        layer_norm_eps
        pre_norm
            Whether to attach the normalization layer before attention layer
            If pre_norm:
                data -> norm(data) -> attn -> res(+data) -> ffn
            Else:
                data -> attn -> norm(res(+data)) -> ffn
        use_qkv_bias
            Whether to use bias for self attention
        activation
            The activation
        layout
            The layout
        """
        super().__init__()
        self._units = units
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._attention_dropout_prob = attention_dropout_prob
        self._hidden_dropout_prob = hidden_dropout_prob
        self._activation_dropout_prob = activation_dropout_prob
        self._pre_norm = pre_norm
        self._layout = layout
        assert layout in ['TN', 'NT'], 'Invalid layout received = {}. ' \
                                       'Only "TN" and "NT" are accepted!'.format(layout)
        assert self._units % self._num_heads == 0, 'units must be divisive by the number of heads'
        self.dropout_layer = nn.Dropout(hidden_dropout_prob)
        self.attn_qkv = nn.Linear(out_features=3 * units, in_features=units, bias=use_qkv_bias)
        self.attention_proj = nn.Linear(out_features=units, in_features=units, bias=True)
        attention_layout = 'NTK' if self._layout == 'NT' else 'TNK'
        self.attention_cell = \
            MultiHeadAttentionCell(
                query_units=self._units,
                num_heads=self._num_heads,
                attention_dropout=self._attention_dropout_prob,
                scaled=True,
                layout=attention_layout
            )
        self.layer_norm = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=units)
        self.ffn = PositionwiseFFN(units=units, hidden_size=hidden_size,
                                   dropout=hidden_dropout_prob,
                                   activation_dropout=activation_dropout_prob,
                                   layer_norm_eps=layer_norm_eps, activation=activation,
                                   pre_norm=pre_norm)

    @property
    def layout(self) -> str:
        return self._layout

    def forward(self, data, attn_mask):
        """
        Parameters
        ----------
        data :
            If layout == 'NT'
                Shape (batch_size, seq_length, C_in)
            Else
                Shape (seq_length, batch_size, C_in)
        attn_mask :
            Shape (batch_size, seq_length, seq_length)
        Returns
        -------
        out :
            If layout == 'NT'
                Shape (batch_size, seq_length, C_out)
            Else
                Shape (seq_length, batch_size, C_out)
        attn_weight :
            Shape (batch_size, seq_length, seq_length)
        """
        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        query, key, value = th.split(self.attn_qkv(data), self._units, dim=-1)
        query = th.reshape(query, query.shape[:2] + (self._num_heads, -1))
        key = th.reshape(key, key.shape[:2] + (self._num_heads, -1))
        value = th.reshape(value, value.shape[:2] + (self._num_heads, -1))
        out, [_, attn_weight] = self.attention_cell(query, key, value, attn_mask)
        out = self.attention_proj(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        out = self.ffn(out)
        return out, attn_weight


class TransformerDecoderLayer(nn.Module):
    def __init__(self, units: int = 512, mem_units: Optional[int] = None, hidden_size: int = 2048,
                 num_heads: int = 8, activation_dropout: float = 0.0, dropout: float = 0.1,
                 attention_dropout: float = 0.1, layer_norm_eps: float = 1E-5,
                 activation: str = 'relu', gated_proj: bool = False, pre_norm: bool = False,
                 use_qkv_bias: bool = True, layout='NT'):
        """
        Parameters
        ----------
        units
        mem_units
            The number of units in the memory. By default, it is initialized to be the
            same as the units.
        hidden_size
        num_heads
        activation_dropout
        dropout
        attention_dropout
        layer_norm_eps
        activation
        gated_proj
        pre_norm
            Whether to apply normalization before the attention layer
        use_qkv_bias
            Whether to use bias for both self attention and contextual attention
        layout
            Layout of the input
        """
        super().__init__()
        self._units = units
        if mem_units is None:
            mem_units = units
        self._mem_units = mem_units
        self._pre_norm = pre_norm
        self._num_heads = num_heads
        self._attention_dropout = attention_dropout
        self._layout = layout
        assert layout in ['TN', 'NT'], 'Invalid layout received = {}. ' \
                                       'Only "TN" and "NT" are accepted!'.format(layout)
        attention_layout = 'NTK' if layout == 'NT' else 'TNK'
        self.dropout_layer = nn.Dropout(dropout)
        if units % num_heads:
            raise ValueError('In Transformer, units should be divided exactly by the number of '
                             'heads. Received units={}, num_heads={}'.format(units, num_heads))
        self.attn_in_qkv = nn.Linear(out_features=3 * units, in_features=units, bias=use_qkv_bias)
        self.self_attention = MultiHeadAttentionCell(query_units=units, num_heads=num_heads,
                                                     attention_dropout=self._attention_dropout,
                                                     layout=attention_layout)
        self.proj_in = nn.Linear(out_features=units, in_features=units, bias=True)
        self.attn_inter_q = nn.Linear(out_features=units, in_features=units, bias=use_qkv_bias)
        self.attn_inter_k = nn.Linear(out_features=units, in_features=mem_units, bias=use_qkv_bias)
        self.attn_inter_v = nn.Linear(out_features=units, in_features=mem_units, bias=use_qkv_bias)
        self.inter_attention = MultiHeadAttentionCell(query_units=units, num_heads=num_heads,
                                                      attention_dropout=self._attention_dropout,
                                                      layout=attention_layout)
        self.proj_inter = nn.Linear(in_features=units, out_features=units, bias=True)
        self.ln_in = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=units)
        self.ln_inter = nn.LayerNorm(eps=layer_norm_eps, normalized_shape=units)
        self.ffn = PositionwiseFFN(units=units, hidden_size=hidden_size, dropout=dropout,
                                   activation_dropout=activation_dropout,
                                   layer_norm_eps=layer_norm_eps, activation=activation,
                                   gated_proj=gated_proj, pre_norm=pre_norm)
        self.init_weights()

    def init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                # TODO, support default initializer
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, PositionwiseFFN):
                module.init_weights()

    @property
    def units(self) -> int:
        return self._units

    @property
    def layout(self) -> str:
        return self._layout

    def forward(self, data, mem, self_causal_mask, mem_attn_mask):
        """
        Parameters
        ----------
        data :
            - layout = 'NT'
                Shape (batch_size, seq_length, C_in)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_in)
        mem :
            - layout = 'NT'
                Shape (batch_size, mem_length, C_mem)
            - layout = 'TN'
                Shape (mem_length, batch_size, C_mem)
        self_causal_mask :
            Shape (batch_size, seq_length, seq_length)
            Mask for the causal self-attention.
            self_causal_mask[i, j, :] masks the elements that token `j` attends to.
            To understand the self-causal attention mask, we can look at the following example:
                       ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
            'I':         1,    0,     0,     0,      0,     0,      0,      0
            'can':       1,    1,     0,     0,      0,     0,      0,      0
            'now':       1,    1,     1,     0,      0,     0,      0,      0
            'use':       1,    1,     1,     1,      0,     0,      0,      0
            'numpy':     1,    1,     1,     1,      1,     0,      0,      0
            'in':        1,    1,     1,     1,      1,     1,      0,      0
            'Gluon@@':   1,    1,     1,     1,      1,     1,      1,      0
            'NLP':       1,    1,     1,     1,      1,     1,      1,      1
        mem_attn_mask :
            Shape (batch_size, seq_length, mem_length)
            Mask between the decoding input and the memory.
                       ['numpy', 'in', 'Gluon@@', 'NLP']
            'I':         1,     1,      1,      1
            'can':       1,     1,      1,      1
            'now':       1,     1,      1,      1
            'use':       1,     1,      1,      1

        Returns
        -------
        out :
            - layout = 'NT'
                Shape (batch_size, seq_length, C_out)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_out)
        """
        # 1. Get the causal self-attention value
        residual = data
        if self._pre_norm:
            data = self.ln_in(data)
        self_query, self_key, self_value = th.split(self.attn_in_qkv(data), self._units, dim=-1)
        out, [_, self_attn_weight] = self.self_attention(
            self_query.reshape((self_query.shape[0], self_query.shape[1], self._num_heads, -1)),
            self_key.reshape((self_key.shape[0], self_key.shape[1], self._num_heads, -1)),
            self_value.reshape((self_value.shape[0], self_value.shape[1], self._num_heads, -1)),
            self_causal_mask)
        out = self.proj_in(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.ln_in(out)
        # 2. Attend to the contextual memory
        data = out
        residual = data
        if self._pre_norm:
            data = self.ln_inter(data)
        out, [_, context_attn_weight] = self.inter_attention(
            th.reshape(self.attn_inter_q(data),
                       (data.shape[0], data.shape[1], self._num_heads, -1)),
            th.reshape(self.attn_inter_k(mem), (mem.shape[0], mem.shape[1], self._num_heads, -1)),
            th.reshape(self.attn_inter_v(mem), (mem.shape[0], mem.shape[1], self._num_heads, -1)),
            mem_attn_mask)
        out = self.proj_inter(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.ln_inter(out)
        # 3. Encode the output via an FFN layer
        out = self.ffn(out)
        return out

    @property
    def state_batch_axis(self):
        if self.layout == 'NT':
            return 0, 0
        else:
            return 1, 1

    def init_states(self, batch_size, device=None, dtype='float32'):
        """Initialize the states required for incremental decoding

        Parameters
        ----------
        batch_size
        device
        dtype

        Returns
        -------
        init_key
            - layout = 'NT'
                Shape (batch_size, 0, N, C_key)
            - layout = 'TN'
                Shape (0, batch_size, N, C_key)
        init_value :
            - layout = 'NT'
                Shape (batch_size, 0, N, C_value)
            - layout = 'TN'
                Shape (0, batch_size, N, C_value)
        """
        dtype = to_torch_dtype(dtype)
        if self.layout == 'NT':
            init_key = th.zeros(
                size=(batch_size, 0, self._num_heads, self._units // self._num_heads),
                device=device, dtype=dtype)
            init_value = th.zeros(
                size=(batch_size, 0, self._num_heads, self._units // self._num_heads),
                device=device, dtype=dtype)
        else:
            init_key = th.zeros(
                size=(0, batch_size, self._num_heads, self._units // self._num_heads),
                device=device, dtype=dtype)
            init_value = th.zeros(
                size=(0, batch_size, self._num_heads, self._units // self._num_heads),
                device=device, dtype=dtype)
        return init_key, init_value

    def incremental_decode(self, data, states, mem, mem_valid_length, mem_attn_mask=None):
        """Incrementally generate the output given the decoder input.

        Parameters
        ----------
        data
            Shape (batch_size, C_in)
        states
            The previous states, contains
            1. layout = 'NT':
                - prev_multi_key
                    Shape (batch_size, prev_seq_length, num_heads, C_key)
                - prev_multi_value
                    Shape (batch_size, prev_seq_length, num_heads, C_value)
            2. layout = 'TN'
                - prev_multi_key
                    Shape (prev_seq_length, batch_size, num_heads, C_key)
                - prev_multi_value
                    Shape (prev_seq_length, batch_size, num_heads, C_value)
        mem
            The memory
            1. layout = 'NT':
                Shape (batch_size, mem_length, C_mem)
            2. layout = 'TN'
                Shape (mem_length, batch_size, C_mem)
        mem_valid_length
            Valid length of the memory
            Shape (batch_size,)
        mem_attn_mask
            The attention mask between data and the memory
            Has shape (batch_size, 1, mem_length)

        Returns
        -------
        out
            Shape (batch_size, C_out)
        updated_states
            - new_key
                Shape (batch_size, prev_seq_length + 1, num_heads, C_key)
            - new_value
                Shape (batch_size, prev_seq_length + 1, num_heads, C_value)
        """
        batch_size = data.shape[0]
        if self.layout == 'NT':
            time_axis = 1
        else:
            time_axis = 0
        data = data.unsqueeze(time_axis)
        residual = data
        if self._pre_norm:
            data = self.ln_in(data)
        # Shape (B, prev_L, #Head, C_K), (B, prev_L, #Head, C_V)
        #  or (prev_L, B, #Head, C_K), (prev_L, B, #Head, C_V)
        prev_key, prev_value = states
        if mem_attn_mask is None:
            mem_attn_mask = gen_mem_attn_mask(mem, mem_valid_length, data, None, layout=self.layout)
        # 1. Get the causal self-attention value, we need to attend to both the current data
        # and the previous stored key/values
        # Shape (B, 1, 3 * num_heads * C_key)
        #  or (1, B, 3 * num_heads * C_key)
        step_qkv = self.attn_in_qkv(data)
        step_query, step_key, step_value = th.split(step_qkv, self._units, dim=-1)
        step_query = th.reshape(
            step_query, shape=(step_query.shape[0], step_query.shape[1], self._num_heads, -1))
        step_key = th.reshape(step_key,
                              shape=(step_key.shape[0], step_key.shape[1], self._num_heads, -1))
        step_value = th.reshape(
            step_value, shape=(step_value.shape[0], step_value.shape[1], self._num_heads, -1))
        new_key = th.cat([prev_key, step_key], dim=time_axis)
        new_value = th.cat([prev_value, step_value], dim=time_axis)
        out, [_, attn_weight] = self.self_attention(step_query, new_key, new_value, None)
        out = self.proj_in(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.ln_in(out)
        # 2. Attend to the contextual memory
        data = out
        residual = data
        if self._pre_norm:
            data = self.ln_inter(data)
        out, _ = self.inter_attention(
            th.reshape(self.attn_inter_q(data),
                       shape=(data.shape[0], data.shape[1], self._num_heads, -1)),
            th.reshape(self.attn_inter_k(mem),
                       shape=(mem.shape[0], mem.shape[1], self._num_heads, -1)),
            th.reshape(self.attn_inter_v(mem),
                       shape=(mem.shape[0], mem.shape[1], self._num_heads, -1)), mem_attn_mask)
        out = self.proj_inter(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.ln_inter(out)
        # 3. Encode the output via an FFN layer
        out = self.ffn(out)
        out = th.reshape(out, shape=(batch_size, -1))
        return out, (new_key, new_value)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, recurrent=False, units=512, mem_units=None, hidden_size=2048,
                 use_qkv_bias=True, num_heads=8, max_shift=None, activation_dropout=0.0,
                 dropout=0.1, attention_dropout=0.1, gated_proj: bool = False, layer_norm_eps=1E-5,
                 data_norm=False, pre_norm=False, activation='relu', layout='NT'):
        super().__init__()
        self._units = units
        self._mem_units = mem_units
        self.num_layers = num_layers
        self.recurrent = recurrent
        self.max_shift = max_shift
        self._data_norm = data_norm
        self._pre_norm = pre_norm
        self._layout = layout
        assert layout in ['TN', 'NT'], 'Invalid layout received = {}. ' \
                                       'Only "TN" and "NT" are accepted!'.format(layout)
        self.dropout_layer = nn.Dropout(dropout)
        if self._data_norm:
            self.ln_data = nn.LayerNorm(units, eps=layer_norm_eps)
        if self._pre_norm:
            self.ln_final = nn.LayerNorm(units, eps=layer_norm_eps)
        # Construct the intermediate layers
        self.layers = nn.ModuleList()
        real_num_layers = 1 if recurrent else num_layers
        for i in range(real_num_layers):
            self.layers.append(
                TransformerDecoderLayer(units=units, mem_units=mem_units, hidden_size=hidden_size,
                                        num_heads=num_heads, activation_dropout=activation_dropout,
                                        use_qkv_bias=use_qkv_bias, dropout=dropout,
                                        gated_proj=gated_proj, attention_dropout=attention_dropout,
                                        layer_norm_eps=layer_norm_eps, activation=activation,
                                        pre_norm=pre_norm, layout=layout))

    @property
    def units(self):
        return self._units

    @property
    def layout(self) -> str:
        return self._layout

    def forward(self, data, valid_length, mem_data, mem_valid_length):
        """Run forward

        Parameters
        ----------
        data
            - layout = 'NT'
                Shape (batch_size, seq_length, C_in)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_in)
        valid_length
            Shape (batch_size,)
        mem_data
            - layout = 'NT'
                Shape (batch_size, mem_length, C_mem)
            - layout = 'TN'
                Shape (mem_length, batch_size, C_mem)
        mem_valid_length
            Shape (batch_size,)

        Returns
        -------
        out
            - layout = 'NT'
                Shape (batch_size, seq_length, C_out)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_out)
        """
        # 1. Embed the data
        out = self.dropout_layer(data)
        if self._data_norm:
            out = self.ln_data(out)
        self_causal_mask = gen_self_attn_mask(data, valid_length, attn_type='causal',
                                              layout=self._layout)
        mem_attn_mask = gen_mem_attn_mask(mem_data, mem_valid_length, data, valid_length,
                                          layout=self._layout)
        for i in range(self.num_layers):
            if self.recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            out = layer(out, mem_data, self_causal_mask, mem_attn_mask)
        if self._pre_norm:
            out = self.ln_final(out)
        return out

    @property
    def state_batch_axis(self):
        ret = []
        for i in range(self.num_layers):
            if self.recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            ret.append(layer.state_batch_axis)
        return ret

    def init_states(self, batch_size, device=None, dtype='float32'):
        """Initialize the states required for incremental decoding

        Parameters
        ----------
        batch_size
            The batch size
        device
            The device
        dtype
            The data type of the states

        Returns
        -------
        states
            A list of states, each includes:
                - init_key
                    - layout = 'NT'
                        Shape (batch_size, 0, N, C_key)
                    - layout = 'TN'
                        Shape (0, batch_size, N, C_key)
                - init_value :
                    - layout = 'NT'
                        Shape (batch_size, 0, N, C_value)
                    - layout = 'TN'
                        Shape (0, batch_size, N, C_value)
        """
        states = []
        for i in range(self.num_layers):
            if self.recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            states.append(layer.init_states(batch_size=batch_size, device=device, dtype=dtype))
        return states

    def incremental_decode(self, data, states, mem, mem_valid_length):
        """Incrementally generate the output given the decoder input.

        Parameters
        ----------
        data
            Shape (batch_size, C_in)
        states
            The previous states, contain a list of
            1. layout = 'NT'
                - prev_multi_key
                    Shape (batch_size, prev_seq_length, num_heads, C_key)
                - prev_multi_value
                    Shape (batch_size, prev_seq_length, num_heads, C_value)
            2. layout = 'TN'
                - prev_multi_key
                    Shape (prev_seq_length, batch_size, num_heads, C_key)
                - prev_multi_value
                    Shape (prev_seq_length, batch_size, num_heads, C_value)
        mem
            The memory
            1. layout = 'NT'
                Shape (batch_size, mem_length, C_mem)
            2. layout = 'TN'
                Shape (mem_length, batch_size, C_mem)
        mem_valid_length
            Valid length of the memory
            Shape (batch_size,)

        Returns
        -------
        out
            Shape (batch_size, C_out)
        new_states
            The updated states, contain a list of
            1. layout = 'NT'
                - new_key
                    Shape (batch_size, prev_seq_length + 1, num_heads, C_key)
                - new_value
                    Shape (prev_seq_length + 1, batch_size, num_heads, C_value)
            2. layout = 'TN'
                - new_key
                    Shape (prev_seq_length + 1, batch_size, num_heads, C_key)
                - new_value
                    Shape (prev_seq_length + 1, batch_size, num_heads, C_value)
        """
        # 1. Embed the data
        out = self.dropout_layer(data)
        if self._data_norm:
            out = self.ln_data(out)
        time_axis = 0 if self.layout == 'TN' else 1
        mem_length = mem.shape[time_axis]
        # Generate the mem_attn_mask
        time_steps = th.arange(mem_length, device=data.device)  # (mem_length,)
        mem_attn_mask = time_steps.reshape((1, 1, -1)) < mem_valid_length.reshape((-1, 1, 1))

        new_states = []
        for i in range(self.num_layers):
            if self.recurrent:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            out, new_state = layer.incremental_decode(out, states[i], mem, mem_valid_length,
                                                      mem_attn_mask)
            new_states.append(new_state)
        if self._pre_norm:
            out = self.ln_final(out)
        return out, new_states
