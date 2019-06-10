import mxnet as mx
import io
import numpy as np
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon import nn
from gluonnlp.model.attention_cell import DotProductAttentionCell
from gluonnlp.model.block import GELU


class GPT2SelfAttentionLayer(Block):
    def __init__(self, units, num_heads, dropout=0.0,
                 weight_initializer=mx.init.Normal(0.02), bias_initializer='zeros', prefix=None, params=None):
        """

        Parameters
        ----------
        units : int
        num_heads : int
        dropout : float
        prefix : str, default None
            Prefix for name of `Block`s
            (and name of weight if params is `None`).
        params : Parameter or None, default None
            Container for weight sharing between cells.
            Created if `None`.
        """
        super(GPT2SelfAttentionLayer, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        assert units % num_heads == 0
        with self.name_scope():
            self._multi_head_qkv_proj = nn.Dense(units=units * 3, flatten=False, use_bias=True,
                                                 weight_initializer=weight_initializer,
                                                 bias_initializer=bias_initializer,
                                                 prefix='qkv_proj_')
            self._base_attn_cell = DotProductAttentionCell(scaled=True, dropout=dropout, prefix='attn_')
            self._dropout_layer = nn.Dropout(dropout)
            self._out_proj = nn.Dense(units=units, flatten=False, use_bias=True,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      prefix='out_proj_')

    def forward(self, data, states=None):
        """

        Parameters
        ----------
        data : mx.nd.NDarray
            The input data, should have shape (batch_size, seq_len, in_dim)
        states : list of NDArray or None
            The states, contains the previous encoded key/values
            prev_key (batch_size, num_heads, past_seq_len, ele_units),
            prev_value (batch_size, num_heads, past_seq_len, ele_units)
            None means no previous states

        Returns
        -------
        out : mx.nd.NDArray
        new_states :
        """
        batch_size = data.shape[0]
        seq_len = data.shape[1]
        # Generate mask
        if states is not None:
            prev_key, prev_value = states
            prev_len = prev_key.shape[2]
        else:
            prev_key, prev_value = None, None
            prev_len = 0
        data_pos = mx.nd.arange(prev_len, prev_len + seq_len, ctx=data.context, dtype=data.dtype)
        all_pos = mx.nd.arange(seq_len + prev_len, ctx=data.context, dtype=data.dtype)
        mask = mx.nd.broadcast_lesser_equal(all_pos.reshape((1, -1)), data_pos.reshape((-1, 1)))
        mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0, size=batch_size * self._num_heads)

        # Multi-head attention
        qkv = self._multi_head_qkv_proj(data)  # Shape (batch_size, seq_len, 3 * units)
        qkv = mx.nd.swapaxes(qkv, 1, 2)  # Shape (batch_size, 3 * units, seq_len)
        query, key, value = mx.nd.split(qkv, num_outputs=3, axis=1)  # Each has shape (batch_size, units, seq_len)
        # Map each to have shape (batch_size * num_head, ele_units, seq_len)
        query = query.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(shape=(-1, 0, 0), reverse=True)
        key = key.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(shape=(-1, 0, 0), reverse=True)
        value = value.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(shape=(-1, 0, 0), reverse=True)
        query = mx.nd.swapaxes(query, 1, 2)
        key = mx.nd.swapaxes(key, 1, 2)
        value = mx.nd.swapaxes(value, 1, 2)
        if prev_key is not None:
            key = mx.nd.concat(prev_key.reshape((-1, 0, 0), reverse=True),
                               key, dim=1)  # Shape (batch_size * num_heads, all_len, ele_units)
        if prev_value is not None:
            value = mx.nd.concat(prev_value.reshape((-1, 0, 0), reverse=True),
                                 value, dim=1)
        out, _ = self._base_attn_cell(query, key, value, mask)  # Shape (batch_size * num_heads, all_len, ele_units)
        out = mx.nd.transpose(out.reshape((-1, self._num_heads, 0, 0), reverse=True),
                              axes=(0, 2, 1, 3)).reshape((0, 0, -1))
        out = self._out_proj(out)
        return out, [key.reshape((-1, self._num_heads, 0, 0), reverse=True),
                     value.reshape((-1, self._num_heads, 0, 0), reverse=True)]


class GPT2FFNLayer(HybridBlock):
    def __init__(self, units, hidden_size,
                 weight_initializer=mx.init.Normal(0.02), bias_initializer='zeros', prefix=None, params=None):
        super(GPT2FFNLayer, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._hidden_size = hidden_size
        with self.name_scope():
            self._hidden_map = nn.Dense(flatten=False, units=hidden_size,
                                        weight_initializer=weight_initializer, bias_initializer=bias_initializer)
            self._out_map = nn.Dense(flatten=False, units=units,
                                     weight_initializer=weight_initializer, bias_initializer=bias_initializer)
            self._act = GELU()

    def hybrid_forward(self, F, data):
        """

        Parameters
        ----------
        F
        data : NDArray or Symbol
            Shape (batch_size, seq_len, in_units)

        Returns
        -------
        out : NDArray or Symbol
            Shape (batch_size, seq_len, units)
        """
        out = self._out_map(self._act(self._hidden_map(data)))
        return out


class GPT2Model(Block):
    def __init__(self, units, vocab_size, max_seq_len, num_layers, num_heads, dropout=0.0,
                 prefix=None, params=None):
        """

        Parameters
        ----------
        units : int
        vocab_size : int
        max_seq_len : int
            The maximum sequence length
        num_layers : int
        num_heads: int
        dropout : float
        prefix : str, default None
            Prefix for name of `Block`s
            (and name of weight if params is `None`).
        params : Parameter or None, default None
            Container for weight sharing between cells.
            Created if `None`.
        """
        super(GPT2Model, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._max_seq_len = max_seq_len
        self._num_layers = num_layers
        self._num_heads = num_heads
        with self.name_scope():
            self._pos_embed = nn.Embedding(input_dim=max_seq_len, output_dim=units, prefix='pos_embed_',
                                           weight_initializer=mx.init.Normal(0.01))
            self._embed = nn.Embedding(input_dim=vocab_size, output_dim=units, prefix='embed_',
                                       weight_initializer=mx.init.Normal(0.02))
            self._logits_proj = nn.Dense(units=vocab_size, in_units=units, use_bias=False, flatten=False,
                                         params=self._embed.params)
            self._self_attention_layers = nn.Sequential()
            self._ffn_layers = nn.HybridSequential()
            self._attn_ln = nn.HybridSequential()
            self._ffn_ln = nn.HybridSequential()
            for i in range(num_layers):
                self._self_attention_layers.add(GPT2SelfAttentionLayer(units=units, num_heads=num_heads,
                                                                       dropout=dropout,
                                                                       prefix='self_attn{}_'.format(i)))
                self._ffn_layers.add(GPT2FFNLayer(units=units, hidden_size=units * 4, prefix='ffn{}_'.format(i)))
                self._attn_ln.add(nn.LayerNorm(prefix='attn_ln{}_'.format(i)))
                self._ffn_ln.add(nn.LayerNorm(prefix='ffn_ln{}_'.format(i)))
                self._final_ln = nn.LayerNorm(prefix='final_ln{}_'.format(i))

    def forward(self, data, states=None):
        """

        Parameters
        ----------
        data : NDArray
            Shape (batch_size, seq_len)
        states : list of NDArray or None

        Returns
        -------
        out : NDArray
            Shape (batch_size, seq_len, vocab_size)
        new_states : list of NDArray
        """
        new_states = []
        batch_size, seq_len = data.shape[0], data.shape[1]
        if states is not None:
            prev_len = states[0].shape[1]
        else:
            prev_len = 0
        assert seq_len + prev_len <= self._max_seq_len
        data_pos = mx.nd.arange(prev_len, prev_len + seq_len, ctx=data.context, dtype=np.float32)
        data_pos =  mx.nd.broadcast_axes(mx.nd.expand_dims(data_pos, axis=0), axis=0, size=batch_size)
        out = self._embed(data) + self._pos_embed(data_pos)
        for i in range(self._num_layers):
            attn_layer = self._self_attention_layers[i]
            ffn_layer = self._ffn_layers[i]
            attn_ln = self._attn_ln[i]
            ffn_ln = self._ffn_ln[i]
            layer_states = None if states is None else states[2*i:(2*i + 2)]
            h, new_layer_states = attn_layer(attn_ln(out), layer_states)
            out = out + h
            h = ffn_layer(ffn_ln(out))
            out = out + h
            new_states.extend(new_layer_states)
        out = self._final_ln(out)
        logits = self._logits_proj(out)
        return logits, new_states


def GPT2_117M():
    return GPT2Model(units=768, vocab_size=50257, max_seq_len=1024, num_heads=12, num_layers=12)


def GPT2_345M():
    return GPT2Model(units=1024, vocab_size=50257, max_seq_len=1024, num_heads=16, num_layers=24)


def load_pretrained_GPT2(model_name='117M', ctx=None):
    """

    Parameters
    ----------
    model_name : str
        Can be 117M or 345M

    Returns
    -------
    model : GPT2Model
    vocab : Vocab
    tokenizer : GPT2Tokenizer
    detokenizer : GPT2Detokenizer
    """
    from gluonnlp.vocab import Vocab
    from transforms import GPT2Tokenizer, GPT2Detokenizer
    if model_name == '117M':
        model = GPT2_117M()
        model.load_parameters(filename='models/117M/model.params', ctx=ctx)
        tokenizer = GPT2Tokenizer(bpe_ranks_path='models/117M/bpe_ranks.json')
        detokenizer = GPT2Detokenizer(tokenizer)
        with io.open('models/117M/vocab.json', 'r', encoding='utf-8') as f:
            vocab = Vocab.from_json(f.read())
    elif model_name == '345M':
        model = GPT2_345M()
        model.load_parameters(filename='models/345M/model.params', ctx=ctx)
        tokenizer = GPT2Tokenizer(bpe_ranks_path='models/345M/bpe_ranks.json')
        detokenizer = GPT2Detokenizer(tokenizer)
        with io.open('models/345M/vocab.json', 'r', encoding='utf-8') as f:
            vocab = Vocab.from_json(f.read())
    else:
        raise NotImplementedError('{} is not found! Try "load_pretrained_GPT2(\'117M\')" or '
                                  '"load_pretrained_GPT2(\'345M\')"')
    for i in range(model._num_layers):
        model._ffn_layers[i]._act._support_erf = False
    return model, vocab, tokenizer, detokenizer
