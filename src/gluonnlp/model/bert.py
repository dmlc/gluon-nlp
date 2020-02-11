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
"""BERT models."""
# pylint: disable=too-many-lines

__all__ = ['BERTModel', 'RoBERTaModel', 'BERTEncoder', 'BERTClassifier',
           'RoBERTaClassifier', 'bert_12_768_12', 'bert_24_1024_16',
           'ernie_12_768_12', 'roberta_12_768_12', 'roberta_24_1024_16',
           'DistilBERTModel', 'distilbert_6_768_12']

import os

import mxnet as mx
from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.model_zoo import model_store

from ..base import get_home_dir
from .block import GELU
from .seq2seq_encoder_decoder import Seq2SeqEncoder
from .transformer import PositionwiseFFN
from .utils import _load_pretrained_params, _load_vocab

###############################################################################
#                              COMPONENTS                                     #
###############################################################################

class DotProductSelfAttentionCell(HybridBlock):
    r"""Multi-head Dot Product Self Attention Cell.

    In the DotProductSelfAttentionCell, the input query/key/value will be linearly projected
    for `num_heads` times with different projection matrices. Each projected key, value, query
    will be used to calculate the attention weights and values. The output of each head will be
    concatenated to form the final output.

    This is a more efficient implementation of MultiHeadAttentionCell with
    DotProductAttentionCell as the base_cell:

    score = <W_q h_q, W_k h_k> / sqrt(dim_q)

    Parameters
    ----------
    units : int
        Total number of projected units for query. Must be divided exactly by num_heads.
    num_heads : int
        Number of parallel attention heads
    use_bias : bool, default True
        Whether to use bias when projecting the query/key/values
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    prefix : str or None, default None
        See document of `Block`.
    params : str or None, default None
        See document of `Block`.

    Inputs:
      - **qkv** : Symbol or NDArray
        Query / Key / Value vector. Shape (query_length, batch_size, C_in)
      - **valid_len** : Symbol or NDArray or None, default None
        Valid length of the query/key/value slots. Shape (batch_size, query_length)

    Outputs:
      - **context_vec** : Symbol or NDArray
        Shape (query_length, batch_size, context_vec_dim)
      - **att_weights** : Symbol or NDArray
        Attention weights of multiple heads.
        Shape (batch_size, num_heads, query_length, memory_length)
    """
    def __init__(self, units, num_heads, dropout=0.0, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._num_heads = num_heads
        self._use_bias = use_bias
        self._dropout = dropout
        self.units = units
        with self.name_scope():
            if self._use_bias:
                self.query_bias = self.params.get('query_bias', shape=(self.units,),
                                                  init=bias_initializer)
                self.key_bias = self.params.get('key_bias', shape=(self.units,),
                                                init=bias_initializer)
                self.value_bias = self.params.get('value_bias', shape=(self.units,),
                                                  init=bias_initializer)
            weight_shape = (self.units, self.units)
            self.query_weight = self.params.get('query_weight', shape=weight_shape,
                                                init=weight_initializer,
                                                allow_deferred_init=True)
            self.key_weight = self.params.get('key_weight', shape=weight_shape,
                                              init=weight_initializer,
                                              allow_deferred_init=True)
            self.value_weight = self.params.get('value_weight', shape=weight_shape,
                                                init=weight_initializer,
                                                allow_deferred_init=True)
            self.dropout_layer = nn.Dropout(self._dropout)

    def _collect_params_with_prefix(self, prefix=''):
        # the registered parameter names in v0.8 are the following:
        # prefix_proj_query.weight, prefix_proj_query.bias
        # prefix_proj_value.weight, prefix_proj_value.bias
        # prefix_proj_key.weight, prefix_proj_key.bias
        # this is a temporary fix to keep backward compatibility, due to an issue in MXNet:
        # https://github.com/apache/incubator-mxnet/issues/17220
        if prefix:
            prefix += '.'
        ret = {prefix + 'proj_' + k.replace('_', '.') : v for k, v in self._reg_params.items()}
        for name, child in self._children.items():
            ret.update(child._collect_params_with_prefix(prefix + name))
        return ret

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, qkv, valid_len, query_bias, key_bias, value_bias,
                       query_weight, key_weight, value_weight):
        # interleaved_matmul_selfatt ops assume the projection is done with interleaving
        # weights for query/key/value. The concatenated weight should have shape
        # (num_heads, C_out/num_heads * 3, C_in).
        query_weight = query_weight.reshape(shape=(self._num_heads, -1, 0), reverse=True)
        key_weight = key_weight.reshape(shape=(self._num_heads, -1, 0), reverse=True)
        value_weight = value_weight.reshape(shape=(self._num_heads, -1, 0), reverse=True)
        in_weight = F.concat(query_weight, key_weight, value_weight, dim=-2)
        in_weight = in_weight.reshape(shape=(-1, 0), reverse=True)
        in_bias = F.concat(query_bias, key_bias, value_bias, dim=0)

        # qkv_proj shape = (seq_length, batch_size, num_heads * head_dim * 3)
        qkv_proj = F.FullyConnected(data=qkv, weight=in_weight, bias=in_bias,
                                    num_hidden=self.units*3, no_bias=False, flatten=False)
        att_score = F.contrib.interleaved_matmul_selfatt_qk(qkv_proj, heads=self._num_heads)
        if valid_len is not None:
            valid_len = F.broadcast_axis(F.expand_dims(valid_len, axis=1),
                                         axis=1, size=self._num_heads)
            valid_len = valid_len.reshape(shape=(-1, 0), reverse=True)
            att_weights = F.softmax(att_score, length=valid_len, use_length=True, axis=-1)
        else:
            att_weights = F.softmax(att_score, axis=-1)
        # att_weights shape = (batch_size, seq_length, seq_length)
        att_weights = self.dropout_layer(att_weights)
        context_vec = F.contrib.interleaved_matmul_selfatt_valatt(qkv_proj, att_weights,
                                                                  heads=self._num_heads)
        att_weights = att_weights.reshape(shape=(-1, self._num_heads, 0, 0), reverse=True)
        return context_vec, att_weights


class BERTEncoderCell(HybridBlock):
    """Structure of the BERT Encoder Cell.

    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    dropout : float
    output_attention: bool
        Whether to output the attention weights
    attention_use_bias : float, default True
        Whether to use bias term in the attention cell
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'gelu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default 1e-5
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence. Shape (length, batch_size, C_in)
        - **valid_length** : valid length of inputs for attention. Shape (batch_size, length)

    Outputs:
        - **outputs**: output tensor of the transformer encoder cell.
            Shape (length, batch_size, C_out)
        - **additional_outputs**: the additional output of all the BERT encoder cell.
    """
    def __init__(self, units=128, hidden_size=512, num_heads=4,
                 dropout=0.0, output_attention=False,
                 attention_use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__(prefix=prefix, params=params)
        self._dropout = dropout
        self._output_attention = output_attention
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.attention_cell = DotProductSelfAttentionCell(units, num_heads,
                                                              use_bias=attention_use_bias,
                                                              dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=True,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer, prefix='proj_')
            self.ffn = PositionwiseFFN(units=units, hidden_size=hidden_size, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer, activation=activation,
                                       layer_norm_eps=layer_norm_eps)
            self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=layer_norm_eps)


    def hybrid_forward(self, F, inputs, valid_len=None):  # pylint: disable=arguments-differ
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (length, batch_size, C_in)
        valid_len : Symbol or NDArray or None
            Valid length for inputs. Shape (batch_size, length)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the transformer encoder cell. Shape (length, batch_size, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        outputs, attention_weights = self.attention_cell(inputs, valid_len)
        outputs = self.proj(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        # use residual
        outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_weights)
        return outputs, additional_outputs

class BERTEncoder(HybridBlock, Seq2SeqEncoder):
    """Structure of the BERT Encoder.

    Different from the original encoder for transformer, `BERTEncoder` uses
    learnable positional embedding, a 'gelu' activation functions and a
    separate epsilon value for LayerNorm.

    Parameters
    ----------
    num_layers : int
        Number of attention layers.
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities and embedding.
    output_attention: bool, default False
        Whether to output the attention weights
    output_all_encodings: bool, default False
        Whether to output encodings of all encoder cells
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None.
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'gelu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default 1e-12
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence of shape (length, batch_size, C_in)
        - **states** : list of tensors for initial states and valid length for self attention.
        - **valid_length** : valid lengths of each sequence. Usually used when part of sequence
            has been padded. Shape is (batch_size, )

    Outputs:
        - **outputs** : the output of the encoder. Shape is (length, batch_size, C_out)
        - **additional_outputs** : list of tensors.
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, num_heads, length, mem_length)

    """

    def __init__(self, *, num_layers=2, units=512, hidden_size=2048,
                 max_length=50, num_heads=4, dropout=0.0,
                 output_attention=False, output_all_encodings=False, weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None, activation='gelu',
                 layer_norm_eps=1e-12):
        super().__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In BERTEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._max_length = max_length
        self._units = units
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._dropout = dropout

        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=1e-12)
            self.position_weight = self.params.get('position_weight', shape=(max_length, units),
                                                   init=weight_initializer)
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                cell = BERTEncoderCell(
                    units=units, hidden_size=hidden_size, num_heads=num_heads,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer, dropout=dropout,
                    output_attention=output_attention, prefix='transformer%d_' % i,
                    activation=activation, layer_norm_eps=layer_norm_eps)
                self.transformer_cells.add(cell)

    def __call__(self, inputs, states=None, valid_length=None):  # pylint: disable=arguments-differ
        """Encode the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray or Symbol
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or Symbols
            Initial states. The list of initial states and valid length for self attention
        valid_length : NDArray or Symbol
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the transformer encoder. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder
        """
        return super().__call__(inputs, states, valid_length)

    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None):
        # pylint: disable=arguments-differ
        """Encode the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray or Symbol
            Input sequence. Shape (length, batch_size, C_in)
        states : list of NDArrays or Symbols
            Initial states. The list of initial states and valid length for self attention
        valid_length : NDArray or Symbol
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        outputs : NDArray or Symbol, or List[NDArray] or List[Symbol]
            If output_all_encodings flag is False, then the output of the last encoder.
            If output_all_encodings flag is True, then the list of all outputs of all encoders.
            In both cases, shape of the tensor(s) is/are (length, batch_size, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length) or
            (batch_size, num_heads, length, length)

        """
        # axis 0 is for length
        steps = F.contrib.arange_like(inputs, axis=0)
        if valid_length is not None:
            zeros = F.zeros_like(steps)
            # valid_length for attention, shape = (batch_size, seq_length)
            attn_valid_len = F.broadcast_add(F.reshape(valid_length, shape=(-1, 1)),
                                             F.reshape(zeros, shape=(1, -1)))
            attn_valid_len = F.cast(attn_valid_len, dtype='int32')
            if states is None:
                states = [attn_valid_len]
            else:
                states.append(attn_valid_len)
        else:
            attn_valid_len = None

        if states is None:
            states = [steps]
        else:
            states.append(steps)

        # positional encoding
        positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
        inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=1))

        if self._dropout:
            inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        outputs = inputs

        all_encodings_outputs = []
        additional_outputs = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, attn_valid_len)
            inputs = outputs
            if self._output_all_encodings:
                if valid_length is not None:
                    outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                             use_sequence_length=True, axis=0)
                all_encodings_outputs.append(outputs)

            if self._output_attention:
                additional_outputs.append(attention_weights)

        if valid_length is not None and not self._output_all_encodings:
            # if self._output_all_encodings, SequenceMask is already applied above
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=0)

        if self._output_all_encodings:
            return all_encodings_outputs, additional_outputs
        return outputs, additional_outputs


###############################################################################
#                                FULL MODEL                                   #
###############################################################################


class BERTModel(HybridBlock):
    """Generic Model for BERT (Bidirectional Encoder Representations from Transformers).

    Parameters
    ----------
    encoder : BERTEncoder
        Bidirectional encoder that encodes the input sentence.
    vocab_size : int or None, default None
        The size of the vocabulary.
    token_type_vocab_size : int or None, default None
        The vocabulary size of token types (number of segments).
    units : int or None, default None
        Number of units for the final pooler layer.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the word and token type
        embeddings if word_embed and token_type_embed are None.
    embed_initializer : Initializer, default None
        Initializer of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    word_embed : Block or None, default None
        The word embedding. If set to None, word_embed will be constructed using embed_size.
    token_type_embed : Block or None, default None
        The token type embedding (segment embedding). If set to None and the token_type_embed will
        be constructed using embed_size.
    use_pooler : bool, default True
        Whether to include the pooler which converts the encoded sequence tensor of shape
        (batch_size, seq_length, units) to a tensor of shape (batch_size, units)
        for segment level classification task.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
    use_classifier : bool, default True
        Whether to include the classifier for next sentence classification.
    use_token_type_embed : bool, default True
        Whether to include token type embedding (segment embedding).
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **token_types**: optional input token type tensor, shape (batch_size, seq_length).
            If the inputs contain two sequences, then the token type of the first
            sequence differs from that of the second one.
        - **valid_length**: optional tensor of input sequence valid lengths, shape (batch_size,)
        - **masked_positions**: optional tensor of position of tokens for masked LM decoding,
            shape (batch_size, num_masked_positions).

    Outputs:
        - **sequence_outputs**: Encoded sequence, which can be either a tensor of the last
            layer of the Encoder, or a list of all sequence encodings of all layers.
            In both cases shape of the tensor(s) is/are (batch_size, seq_length, units).
        - **attention_outputs**: output list of all intermediate encodings per layer
            Returned only if BERTEncoder.output_attention is True.
            List of num_layers length of tensors of shape
            (batch_size, num_attention_heads, seq_length, seq_length)
        - **pooled_output**: output tensor of pooled representation of the first tokens.
            Returned only if use_pooler is True. Shape (batch_size, units)
        - **next_sentence_classifier_output**: output tensor of next sentence classification.
            Returned only if use_classifier is True. Shape (batch_size, 2)
        - **masked_lm_outputs**: output tensor of sequence decoding for masked language model
            prediction. Returned only if use_decoder True.
            Shape (batch_size, num_masked_positions, vocab_size)
    """

    def __init__(self, encoder, vocab_size=None, token_type_vocab_size=None, units=None,
                 embed_size=None, embed_initializer=None,
                 word_embed=None, token_type_embed=None, use_pooler=True, use_decoder=True,
                 use_classifier=True, use_token_type_embed=True, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._use_decoder = use_decoder
        self._use_classifier = use_classifier
        self._use_pooler = use_pooler
        self._use_token_type_embed = use_token_type_embed
        self._units = units
        self.encoder = encoder
        # Construct word embedding
        self.word_embed = self._get_embed(word_embed, vocab_size, embed_size,
                                          embed_initializer, 'word_embed_')
        # Construct token type embedding
        if use_token_type_embed:
            self.token_type_embed = self._get_embed(token_type_embed, token_type_vocab_size,
                                                    embed_size, embed_initializer,
                                                    'token_type_embed_')
        if self._use_pooler:
            # Construct pooler
            self.pooler = self._get_pooler(units, 'pooler_')
            if self._use_classifier:
                # Construct classifier for next sentence predicition
                self.classifier = self._get_classifier('cls_')
        else:
            assert not use_classifier, 'Cannot use classifier if use_pooler is False'
        if self._use_decoder:
            # Construct decoder for masked language model
            self.decoder = self._get_decoder(units, vocab_size, self.word_embed[0], 'decoder_')

    def _get_classifier(self, prefix):
        """ Construct a decoder for the next sentence prediction task """
        with self.name_scope():
            classifier = nn.Dense(2, prefix=prefix)
        return classifier

    def _get_decoder(self, units, vocab_size, embed, prefix):
        """ Construct a decoder for the masked language model task """
        with self.name_scope():
            decoder = nn.HybridSequential(prefix=prefix)
            decoder.add(nn.Dense(units, flatten=False))
            decoder.add(GELU())
            decoder.add(nn.LayerNorm(in_channels=units, epsilon=1e-12))
            decoder.add(nn.Dense(vocab_size, flatten=False, params=embed.collect_params()))
        assert decoder[3].weight == list(embed.collect_params().values())[0], \
            'The weights of word embedding are not tied with those of decoder'
        return decoder

    def _get_embed(self, embed, vocab_size, embed_size, initializer, prefix):
        """ Construct an embedding block. """
        if embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "word_embed" or ' \
                                           'token_type_embed is not given.'
            with self.name_scope():
                embed = nn.HybridSequential(prefix=prefix)
                with embed.name_scope():
                    embed.add(nn.Embedding(input_dim=vocab_size, output_dim=embed_size,
                                           weight_initializer=initializer))
        assert isinstance(embed, HybridBlock)
        return embed

    def _get_pooler(self, units, prefix):
        """ Construct pooler.

        The pooler slices and projects the hidden output of first token
        in the sequence for segment level classification.

        """
        with self.name_scope():
            pooler = nn.Dense(units=units, flatten=False, activation='tanh',
                              prefix=prefix)
        return pooler

    def __call__(self, inputs, token_types, valid_length=None, masked_positions=None):
        # pylint: disable=dangerous-default-value, arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        return super().__call__(inputs, token_types, valid_length, masked_positions)

    def hybrid_forward(self, F, inputs, token_types, valid_length=None, masked_positions=None):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        outputs = []
        seq_out, attention_out = self._encode_sequence(inputs, token_types, valid_length)
        outputs.append(seq_out)

        if self.encoder._output_all_encodings:
            assert isinstance(seq_out, list)
            output = seq_out[-1]
        else:
            output = seq_out

        if attention_out:
            outputs.append(attention_out)

        if self._use_pooler:
            pooled_out = self._apply_pooling(output)
            outputs.append(pooled_out)
            if self._use_classifier:
                next_sentence_classifier_out = self.classifier(pooled_out)
                outputs.append(next_sentence_classifier_out)
        if self._use_decoder:
            assert masked_positions is not None, \
                'masked_positions tensor is required for decoding masked language model'
            decoder_out = self._decode(F, output, masked_positions)
            outputs.append(decoder_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def _encode_sequence(self, inputs, token_types, valid_length=None):
        """Generate the representation given the input sequences.

        This is used for pre-training or fine-tuning a BERT model.
        """
        # embedding
        embedding = self.word_embed(inputs)
        if self._use_token_type_embed:
            type_embedding = self.token_type_embed(token_types)
            embedding = embedding + type_embedding
        # (batch, seq_len, C) -> (seq_len, batch, C)
        embedding = embedding.transpose((1, 0, 2))
        # encoding
        outputs, additional_outputs = self.encoder(embedding, valid_length=valid_length)
        # (seq_len, batch, C) -> (batch, seq_len, C)
        if isinstance(outputs, (list, tuple)):
            outputs = [o.transpose((1, 0, 2)) for o in outputs]
        else:
            outputs = outputs.transpose((1, 0, 2))
        return outputs, additional_outputs

    def _apply_pooling(self, sequence):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a BERT model.
        """
        outputs = sequence.slice(begin=(0, 0, 0), end=(None, 1, None))
        outputs = outputs.reshape(shape=(-1, self._units))
        return self.pooler(outputs)

    def _decode(self, F, sequence, masked_positions):
        """Generate unnormalized prediction for the masked language model task.

        This is only used for pre-training the BERT model.

        Inputs:
            - **sequence**: input tensor of sequence encodings.
              Shape (batch_size, seq_length, units).
            - **masked_positions**: input tensor of position of tokens for masked LM decoding.
              Shape (batch_size, num_masked_positions). For each sample in the batch, the values
              in this tensor must not be out of bound considering the length of the sequence.

        Outputs:
            - **masked_lm_outputs**: output tensor of token predictions for target masked_positions.
                Shape (batch_size, num_masked_positions, vocab_size).
        """
        masked_positions = masked_positions.astype('int32')
        mask_shape = masked_positions.shape_array()
        num_masked_positions = mask_shape.slice(begin=(1,), end=(2,)).astype('int32')
        idx_arange = F.contrib.arange_like(masked_positions.reshape((-1, )), axis=0)
        batch_idx = F.broadcast_div(idx_arange, num_masked_positions)
        # batch_idx_1d =        [0,0,0,1,1,1,2,2,2...]
        # masked_positions_1d = [1,2,4,0,3,4,2,3,5...]
        batch_idx_1d = batch_idx.reshape((1, -1))
        masked_positions_1d = masked_positions.reshape((1, -1))
        position_idx = F.concat(batch_idx_1d, masked_positions_1d, dim=0)
        encoded = F.gather_nd(sequence, position_idx)
        encoded = encoded.reshape_like(masked_positions, lhs_begin=-2, lhs_end=-1, rhs_begin=0)
        decoded = self.decoder(encoded)
        return decoded


class RoBERTaModel(BERTModel):
    """Generic Model for BERT (Bidirectional Encoder Representations from Transformers).

    Parameters
    ----------
    encoder : BERTEncoder
        Bidirectional encoder that encodes the input sentence.
    vocab_size : int or None, default None
        The size of the vocabulary.
    units : int or None, default None
        Number of units for the final pooler layer.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the word and token type
        embeddings if word_embed and token_type_embed are None.
    embed_initializer : Initializer, default None
        Initializer of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    word_embed : Block or None, default None
        The word embedding. If set to None, word_embed will be constructed using embed_size.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **valid_length**: optional tensor of input sequence valid lengths, shape (batch_size,)
        - **masked_positions**: optional tensor of position of tokens for masked LM decoding,
            shape (batch_size, num_masked_positions).

    Outputs:
        - **sequence_outputs**: Encoded sequence, which can be either a tensor of the last
            layer of the Encoder, or a list of all sequence encodings of all layers.
            In both cases shape of the tensor(s) is/are (batch_size, seq_length, units).
        - **attention_outputs**: output list of all intermediate encodings per layer
            Returned only if BERTEncoder.output_attention is True.
            List of num_layers length of tensors of shape
            (num_masks, num_attention_heads, seq_length, seq_length)
        - **masked_lm_outputs**: output tensor of sequence decoding for masked language model
            prediction. Returned only if use_decoder True.
            Shape (batch_size, num_masked_positions, vocab_size)
    """

    def __init__(self, encoder, vocab_size=None, units=None,
                 embed_size=None, embed_initializer=None,
                 word_embed=None, use_decoder=True,
                 prefix=None, params=None):
        super(RoBERTaModel, self).__init__(encoder, vocab_size=vocab_size,
                                           token_type_vocab_size=None, units=units,
                                           embed_size=embed_size,
                                           embed_initializer=embed_initializer,
                                           word_embed=word_embed, token_type_embed=None,
                                           use_pooler=False, use_decoder=use_decoder,
                                           use_classifier=False, use_token_type_embed=False,
                                           prefix=prefix, params=params)

    def __call__(self, inputs, valid_length=None, masked_positions=None):
        # pylint: disable=dangerous-default-value
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        return super(RoBERTaModel, self).__call__(inputs, [], valid_length=valid_length,
                                                  masked_positions=masked_positions)

class DistilBERTModel(BERTModel):
    """DistilBERT Model.

    Parameters
    ----------
    encoder : BERTEncoder
        Bidirectional encoder that encodes the input sentence.
    vocab_size : int or None, default None
        The size of the vocabulary.
    units : int or None, default None
        Number of units for the final pooler layer.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the word and token type
        embeddings if word_embed and token_type_embed are None.
    embed_initializer : Initializer, default None
        Initializer of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    word_embed : Block or None, default None
        The word embedding. If set to None, word_embed will be constructed using embed_size.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **valid_length**: optional tensor of input sequence valid lengths, shape (batch_size,)

    Outputs:
        - **sequence_outputs**: Encoded sequence, which can be either a tensor of the last
            layer of the Encoder, or a list of all sequence encodings of all layers.
            In both cases shape of the tensor(s) is/are (batch_size, seq_length, units).
        - **attention_outputs**: output list of all intermediate encodings per layer
            Returned only if BERTEncoder.output_attention is True.
            List of num_layers length of tensors of shape
            (num_masks, num_attention_heads, seq_length, seq_length)
    """

    def __init__(self, encoder, vocab_size=None, units=None,
                 embed_size=None, embed_initializer=None,
                 word_embed=None, prefix=None, params=None):
        super(DistilBERTModel, self).__init__(encoder, vocab_size=vocab_size,
                                              token_type_vocab_size=None, units=units,
                                              embed_size=embed_size,
                                              embed_initializer=embed_initializer,
                                              word_embed=word_embed, token_type_embed=None,
                                              use_pooler=False, use_decoder=False,
                                              use_classifier=False, use_token_type_embed=False,
                                              prefix=prefix, params=params)

    def __call__(self, inputs, valid_length=None):
        # pylint: disable=dangerous-default-value, signature-differs
        """Generate the representation given the inputs.

        This is used in fine-tuning a DistilBERT model.
        """
        return super(DistilBERTModel, self).__call__(inputs, [], valid_length=valid_length)

class BERTClassifier(HybridBlock):
    """Model for sentence (pair) classification task with BERT.

    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for
    classification.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    num_classes : int, default is 2
        The number of target classes.
    dropout : float or None, default 0.0.
        Dropout probability for the bert output.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, bert, num_classes=2, dropout=0.0,
                 prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def __call__(self, inputs, token_types, valid_length=None):
        # pylint: disable=dangerous-default-value, arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray or Symbol, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray or Symbol, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or Symbol, or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray or Symbol
            Shape (batch_size, num_classes)
        """
        return super(BERTClassifier, self).__call__(inputs, token_types, valid_length)

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray or Symbol, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray or Symbol, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, num_classes)
        """
        _, pooler_out = self.bert(inputs, token_types, valid_length)
        return self.classifier(pooler_out)


class RoBERTaClassifier(HybridBlock):
    """Model for sentence (pair) classification task with BERT.

    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for
    classification.

    Parameters
    ----------
    bert: RoBERTaModel
        The RoBERTa model.
    num_classes : int, default is 2
        The number of target classes.
    dropout : float or None, default 0.0.
        Dropout probability for the RoBERTa output.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **valid_length**: optional tensor of input sequence valid lengths.
            Shape (batch_size, num_classes).

    Outputs:
        - **output**: Regression output, shape (batch_size, num_classes)
    """

    def __init__(self, roberta, num_classes=2, dropout=0.0,
                 prefix=None, params=None):
        super(RoBERTaClassifier, self).__init__(prefix=prefix, params=params)
        self.roberta = roberta
        self._units = roberta._units
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=self._units, activation='tanh'))
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def __call__(self, inputs, valid_length=None):
        # pylint: disable=dangerous-default-value, arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray or Symbol, shape (batch_size, seq_length)
            Input words for the sequences.
        valid_length : NDArray or Symbol, or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray or Symbol
            Shape (batch_size, num_classes)
        """
        return super(RoBERTaClassifier, self).__call__(inputs, valid_length)

    def hybrid_forward(self, F, inputs, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray or Symbol, shape (batch_size, seq_length)
            Input words for the sequences.
        valid_length : NDArray or Symbol, or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray or Symbol
            Shape (batch_size, num_classes)
        """
        seq_out = self.roberta(inputs, valid_length)
        assert not isinstance(seq_out, (tuple, list)), 'Expected one output from RoBERTaModel'
        outputs = seq_out.slice(begin=(0, 0, 0), end=(None, 1, None))
        outputs = outputs.reshape(shape=(-1, self._units))
        return self.classifier(outputs)

###############################################################################
#                               GET MODEL                                     #
###############################################################################


model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('5656dac6965b5054147b0375337d5a6a7a2ff832', 'bert_12_768_12_book_corpus_wiki_en_cased'),
        ('75cc780f085e8007b3bf6769c6348bb1ff9a3074', 'bert_12_768_12_book_corpus_wiki_en_uncased'),
        ('e0864cc40b3d00fcfb1a878a728650d9148c9a1d',
         'distilbert_6_768_12_distilbert_book_corpus_wiki_en_uncased'),
        ('a56e24015a777329c795eed4ed21c698af03c9ff',
         'bert_12_768_12_openwebtext_book_corpus_wiki_en_uncased'),
        ('5cf21fcddb5ae1a4c21c61201643460c9d65d3b0',
         'roberta_12_768_12_openwebtext_ccnews_stories_books_cased'),
        ('d1b7163e9628e2fd51c9a9f3a0dc519d4fc24add',
         'roberta_24_1024_16_openwebtext_ccnews_stories_books_cased'),
        ('237f39851b24f0b56d70aa20efd50095e3926e26', 'bert_12_768_12_wiki_multilingual_uncased'),
        ('b0f57a207f85a7d361bb79de80756a8c9a4276f7', 'bert_12_768_12_wiki_multilingual_cased'),
        ('885ebb9adc249a170c5576e90e88cfd1bbd98da6', 'bert_12_768_12_wiki_cn_cased'),
        ('4e685a966f8bf07d533bd6b0e06c04136f23f620', 'bert_24_1024_16_book_corpus_wiki_en_cased'),
        ('24551e1446180e045019a87fc4ffbf714d99c0b5', 'bert_24_1024_16_book_corpus_wiki_en_uncased'),
        ('6c82d963fc8fa79c35dd6cb3e1725d1e5b6aa7d7', 'bert_12_768_12_scibert_scivocab_uncased'),
        ('adf9c81e72ac286a37b9002da8df9e50a753d98b', 'bert_12_768_12_scibert_scivocab_cased'),
        ('75acea8e8386890120533d6c0032b0b3fcb2d536', 'bert_12_768_12_scibert_basevocab_uncased'),
        ('8e86e5de55d6dae99123312cd8cdd8183a75e057', 'bert_12_768_12_scibert_basevocab_cased'),
        ('a07780385add682f609772e81ec64aca77c9fb05', 'bert_12_768_12_biobert_v1.0_pmc_cased'),
        ('280ad1cc487db90489f86189e045e915b35e7489', 'bert_12_768_12_biobert_v1.0_pubmed_cased'),
        ('8a8c75441f028a6b928b11466f3d30f4360dfff5',
         'bert_12_768_12_biobert_v1.0_pubmed_pmc_cased'),
        ('55f15c5d23829f6ee87622b68711b15fef50e55b', 'bert_12_768_12_biobert_v1.1_pubmed_cased'),
        ('60281c98ba3572dfdaac75131fa96e2136d70d5c', 'bert_12_768_12_clinicalbert_uncased'),
        ('f869f3f89e4237a769f1b7edcbdfe8298b480052', 'ernie_12_768_12_baidu_ernie_uncased'),
        ('ccf0593e03b91b73be90c191d885446df935eb64', 'bert_12_768_12_kobert_news_wiki_ko_cased')
    ]})

roberta_12_768_12_hparams = {
    'num_layers': 12,
    'units': 768,
    'hidden_size': 3072,
    'max_length': 512,
    'num_heads': 12,
    'dropout': 0.1,
    'embed_size': 768,
    'word_embed': None,
    'layer_norm_eps': 1e-5
}

roberta_24_1024_16_hparams = {
    'num_layers': 24,
    'units': 1024,
    'hidden_size': 4096,
    'max_length': 512,
    'num_heads': 16,
    'dropout': 0.1,
    'embed_size': 1024,
    'word_embed': None,
    'layer_norm_eps': 1e-5
}

distilbert_6_768_12_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 6,
    'units': 768,
    'hidden_size': 3072,
    'max_length': 512,
    'num_heads': 12,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 768,
    'word_embed': None,
}

bert_12_768_12_hparams = {
    'num_layers': 12,
    'units': 768,
    'hidden_size': 3072,
    'max_length': 512,
    'num_heads': 12,
    'dropout': 0.1,
    'embed_size': 768,
    'token_type_vocab_size': 2,
    'word_embed': None,
}

bert_24_1024_16_hparams = {
    'num_layers': 24,
    'units': 1024,
    'hidden_size': 4096,
    'max_length': 512,
    'num_heads': 16,
    'dropout': 0.1,
    'embed_size': 1024,
    'token_type_vocab_size': 2,
    'word_embed': None,
}

ernie_12_768_12_hparams = {
    'num_layers': 12,
    'units': 768,
    'hidden_size': 3072,
    'max_length': 513,
    'num_heads': 12,
    'dropout': 0.1,
    'embed_size': 768,
    'token_type_vocab_size': 2,
    'word_embed': None,
    'activation': 'relu',
    'layer_norm_eps': 1e-5
}

bert_hparams = {
    'distilbert_6_768_12': distilbert_6_768_12_hparams,
    'bert_12_768_12': bert_12_768_12_hparams,
    'bert_24_1024_16': bert_24_1024_16_hparams,
    'roberta_12_768_12': roberta_12_768_12_hparams,
    'roberta_24_1024_16': roberta_24_1024_16_hparams,
    'ernie_12_768_12': ernie_12_768_12_hparams
}


def bert_12_768_12(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                   root=os.path.join(get_home_dir(), 'models'), use_pooler=True, use_decoder=True,
                   use_classifier=True, pretrained_allow_missing=False,
                   hparam_allow_override=False, **kwargs):
    """Generic BERT BASE model.

    The number of layers (L) is 12, number of units (H) is 768, and the
    number of self-attention heads (A) is 12.

    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        The supported datasets are 'book_corpus_wiki_en_cased',
        'book_corpus_wiki_en_uncased', 'wiki_cn_cased',
        'openwebtext_book_corpus_wiki_en_uncased',
        'wiki_multilingual_uncased', 'wiki_multilingual_cased',
        'scibert_scivocab_uncased', 'scibert_scivocab_cased',
        'scibert_basevocab_uncased', 'scibert_basevocab_cased',
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed',
        'clinicalbert',
        'kobert_news_wiki_ko_cased'
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    use_pooler : bool, default True
        Whether to include the pooler which converts the encoded sequence tensor of shape
        (batch_size, seq_length, units) to a tensor of shape (batch_size, units)
        for for segment level classification task.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
        Note that
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed',
        'clinicalbert'
        do not include these parameters.
    use_classifier : bool, default True
        Whether to include the classifier for next sentence classification.
        Note that
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed'
        do not include these parameters.
    pretrained_allow_missing : bool, default False
        Whether to ignore if any parameters for the BERTModel are missing in
        the pretrained weights for model.
        Some BERTModels for example do not provide decoder or classifier
        weights. In that case it is still possible to construct a BERTModel
        with use_decoder=True and/or use_classifier=True, but the respective
        parameters will be missing from the pretrained file.
        If pretrained_allow_missing=True, this will be ignored and the
        parameters will be left uninitialized. Otherwise AssertionError is
        raised.
    hparam_allow_override : bool, default False
        If set to True, pre-defined hyper-parameters of the model
        (e.g. the number of layers, hidden units) can be overriden.

    The pretrained parameters for dataset_name
    'openwebtext_book_corpus_wiki_en_uncased' were obtained by running the
    GluonNLP BERT pre-training script on OpenWebText.

    The pretrained parameters for dataset_name 'scibert_scivocab_uncased',
    'scibert_scivocab_cased', 'scibert_basevocab_uncased',
    'scibert_basevocab_cased' were obtained by converting the parameters
    published by "Beltagy, I., Cohan, A., & Lo, K. (2019). Scibert: Pretrained
    contextualized embeddings for scientific text. arXiv preprint
    arXiv:1903.10676."

    The pretrained parameters for dataset_name 'biobert_v1.0_pmc',
    'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc', 'biobert_v1.1_pubmed'
    were obtained by converting the parameters published by "Lee, J., Yoon, W.,
    Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2019). Biobert:
    pre-trained biomedical language representation model for biomedical text
    mining. arXiv preprint arXiv:1901.08746."

    The pretrained parameters for dataset_name 'clinicalbert' were obtained by
    converting the parameters published by "Huang, K., Altosaar, J., &
    Ranganath, R. (2019). ClinicalBERT: Modeling Clinical Notes and Predicting
    Hospital Readmission. arXiv preprint arXiv:1904.05342."


    Returns
    -------
    BERTModel, gluonnlp.vocab.BERTVocab
    """
    return get_bert_model(model_name='bert_12_768_12', vocab=vocab, dataset_name=dataset_name,
                          pretrained=pretrained, ctx=ctx, use_pooler=use_pooler,
                          use_decoder=use_decoder, use_classifier=use_classifier, root=root,
                          pretrained_allow_missing=pretrained_allow_missing,
                          hparam_allow_override=hparam_allow_override, **kwargs)


def bert_24_1024_16(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(), use_pooler=True,
                    use_decoder=True, use_classifier=True,
                    root=os.path.join(get_home_dir(), 'models'),
                    pretrained_allow_missing=False,
                    hparam_allow_override=False, **kwargs):
    """Generic BERT LARGE model.

    The number of layers (L) is 24, number of units (H) is 1024, and the
    number of self-attention heads (A) is 16.

    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        Options include 'book_corpus_wiki_en_uncased' and 'book_corpus_wiki_en_cased'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    use_pooler : bool, default True
        Whether to include the pooler which converts the encoded sequence tensor of shape
        (batch_size, seq_length, units) to a tensor of shape (batch_size, units)
        for for segment level classification task.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
    use_classifier : bool, default True
        Whether to include the classifier for next sentence classification.
    pretrained_allow_missing : bool, default False
        Whether to ignore if any parameters for the BERTModel are missing in
        the pretrained weights for model.
        Some BERTModels for example do not provide decoder or classifier
        weights. In that case it is still possible to construct a BERTModel
        with use_decoder=True and/or use_classifier=True, but the respective
        parameters will be missing from the pretrained file.
        If pretrained_allow_missing=True, this will be ignored and the
        parameters will be left uninitialized. Otherwise AssertionError is
        raised.
    hparam_allow_override : bool, default False
        If set to True, pre-defined hyper-parameters of the model
        (e.g. the number of layers, hidden units) can be overriden.

    Returns
    -------
    BERTModel, gluonnlp.vocab.BERTVocab
    """
    return get_bert_model(model_name='bert_24_1024_16', vocab=vocab, dataset_name=dataset_name,
                          pretrained=pretrained, ctx=ctx, use_pooler=use_pooler,
                          use_decoder=use_decoder, use_classifier=use_classifier, root=root,
                          pretrained_allow_missing=pretrained_allow_missing,
                          hparam_allow_override=hparam_allow_override, **kwargs)

def distilbert_6_768_12(dataset_name='distil_book_corpus_wiki_en_uncased', vocab=None,
                        pretrained=True, ctx=mx.cpu(),
                        output_attention=False,
                        output_all_encodings=False,
                        root=os.path.join(get_home_dir(), 'models'),
                        **kwargs):
    """DistilBERT model: https://arxiv.org/abs/1910.01108

    The number of layers (L) is 6, number of units (H) is 768, and the
    number of self-attention heads (A) is 12.

    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        Options include 'book_corpus_wiki_en_uncased' and 'book_corpus_wiki_en_cased'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.

    Returns
    -------
    DistilBERTModel, gluonnlp.vocab.Vocab
    """
    model_name = 'distilbert_6_768_12'
    predefined_args = bert_hparams[model_name]
    mutable_args = ['use_residual', 'dropout', 'word_embed']
    mutable_args = frozenset(mutable_args)
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
        'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    # encoder
    encoder = BERTEncoder(num_layers=predefined_args['num_layers'],
                          units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'],
                          dropout=predefined_args['dropout'],
                          output_attention=output_attention,
                          output_all_encodings=output_all_encodings,
                          activation=predefined_args.get('activation', 'gelu'),
                          layer_norm_eps=predefined_args.get('layer_norm_eps', 1e-5))

    from ..vocab import Vocab  # pylint: disable=import-outside-toplevel
    bert_vocab = _load_vocab(dataset_name, vocab, root, cls=Vocab)
    # DistilBERT
    net = DistilBERTModel(encoder, len(bert_vocab),
                          units=predefined_args['units'],
                          embed_size=predefined_args['embed_size'],
                          word_embed=predefined_args['word_embed'])
    if pretrained:
        _load_pretrained_params(net, model_name, dataset_name, root, ctx,
                                allow_missing=False)
    return net, bert_vocab

def roberta_12_768_12(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                      use_decoder=True,
                      root=os.path.join(get_home_dir(), 'models'),
                      hparam_allow_override=False, **kwargs):
    """Generic RoBERTa BASE model.

    The number of layers (L) is 12, number of units (H) is 768, and the
    number of self-attention heads (A) is 12.

    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        Options include 'book_corpus_wiki_en_uncased' and 'book_corpus_wiki_en_cased'.
    vocab : gluonnlp.vocab.Vocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
    hparam_allow_override : bool, default False
        If set to True, pre-defined hyper-parameters of the model
        (e.g. the number of layers, hidden units) can be overriden.

    Returns
    -------
    RoBERTaModel, gluonnlp.vocab.Vocab
    """
    return get_roberta_model(model_name='roberta_12_768_12', vocab=vocab, dataset_name=dataset_name,
                             pretrained=pretrained, ctx=ctx,
                             use_decoder=use_decoder, root=root,
                             hparam_allow_override=hparam_allow_override, **kwargs)


def roberta_24_1024_16(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                       use_decoder=True,
                       root=os.path.join(get_home_dir(), 'models'),
                       hparam_allow_override=False, **kwargs):
    """Generic RoBERTa LARGE model.

    The number of layers (L) is 24, number of units (H) is 1024, and the
    number of self-attention heads (A) is 16.

    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        Options include 'book_corpus_wiki_en_uncased' and 'book_corpus_wiki_en_cased'.
    vocab : gluonnlp.vocab.Vocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
    hparam_allow_override : bool, default False
        If set to True, pre-defined hyper-parameters of the model
        (e.g. the number of layers, hidden units) can be overriden.

    Returns
    -------
    RoBERTaModel, gluonnlp.vocab.Vocab
    """
    return get_roberta_model(model_name='roberta_24_1024_16', vocab=vocab,
                             dataset_name=dataset_name, pretrained=pretrained, ctx=ctx,
                             use_decoder=use_decoder, root=root,
                             hparam_allow_override=hparam_allow_override, **kwargs)


def ernie_12_768_12(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                    root=os.path.join(get_home_dir(), 'models'), use_pooler=True, use_decoder=True,
                    use_classifier=True, hparam_allow_override=False, **kwargs):
    """Baidu ERNIE model.

    Reference:
    https://arxiv.org/pdf/1904.09223.pdf

    The number of layers (L) is 12, number of units (H) is 768, and the
    number of self-attention heads (A) is 12.

    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        The supported datasets are 'baidu_ernie'
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    use_pooler : bool, default True
        Whether to include the pooler which converts the encoded sequence tensor of shape
        (batch_size, seq_length, units) to a tensor of shape (batch_size, units)
        for for segment level classification task.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
    use_classifier : bool, default True
        Whether to include the classifier for next sentence classification.
    hparam_allow_override : bool, default False
        If set to True, pre-defined hyper-parameters of the model
        (e.g. the number of layers, hidden units) can be overriden.

    Returns
    -------
    (BERTModel, gluonnlp.vocab.BERTVocab)
    """
    return get_bert_model(model_name='ernie_12_768_12', vocab=vocab, dataset_name=dataset_name,
                          pretrained=pretrained, ctx=ctx, use_pooler=use_pooler,
                          use_decoder=use_decoder, use_classifier=use_classifier, root=root,
                          pretrained_allow_missing=False,
                          hparam_allow_override=hparam_allow_override, **kwargs)


def get_roberta_model(model_name=None, dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                      use_decoder=True, output_attention=False, output_all_encodings=False,
                      root=os.path.join(get_home_dir(), 'models'), ignore_extra=False,
                      hparam_allow_override=False, **kwargs):
    """Any RoBERTa pretrained model.

    Parameters
    ----------
    model_name : str or None, default None
        Options include 'bert_24_1024_16' and 'bert_12_768_12'.
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        The supported datasets for model_name of either roberta_24_1024_16 and
        roberta_12_768_12 include 'openwebtext_ccnews_stories_books'.
    vocab : gluonnlp.vocab.Vocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
        Note that
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed',
        'clinicalbert'
        do not include these parameters.
    output_attention : bool, default False
        Whether to include attention weights of each encoding cell to the output.
    output_all_encodings : bool, default False
        Whether to output encodings of all encoder cells.
    ignore_extra : bool, default False
        Whether to silently ignore parameters from the file that are not
        present in this Block.
    hparam_allow_override : bool, default False
        If set to True, pre-defined hyper-parameters of the model
        (e.g. the number of layers, hidden units) can be overriden.

    Returns
    -------
    RoBERTaModel, gluonnlp.vocab.Vocab
    """
    predefined_args = bert_hparams[model_name].copy()
    if not hparam_allow_override:
        mutable_args = ['use_residual', 'dropout', 'word_embed']
        mutable_args = frozenset(mutable_args)
        assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
            'Cannot override predefined model settings.'
    predefined_args.update(kwargs)

    # encoder
    encoder = BERTEncoder(num_layers=predefined_args['num_layers'],
                          units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'],
                          dropout=predefined_args['dropout'],
                          output_attention=output_attention,
                          output_all_encodings=output_all_encodings,
                          activation=predefined_args.get('activation', 'gelu'),
                          layer_norm_eps=predefined_args.get('layer_norm_eps', 1e-5))

    from ..vocab import Vocab  # pylint: disable=import-outside-toplevel
    bert_vocab = _load_vocab(dataset_name, vocab, root, cls=Vocab)
    # BERT
    net = RoBERTaModel(encoder, len(bert_vocab),
                       units=predefined_args['units'],
                       embed_size=predefined_args['embed_size'],
                       word_embed=predefined_args['word_embed'],
                       use_decoder=use_decoder)
    if pretrained:
        ignore_extra = ignore_extra or not use_decoder
        _load_pretrained_params(net, model_name, dataset_name, root, ctx, ignore_extra=ignore_extra,
                                allow_missing=False)
    return net, bert_vocab


def get_bert_model(model_name=None, dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                   use_pooler=True, use_decoder=True, use_classifier=True, output_attention=False,
                   output_all_encodings=False, use_token_type_embed=True,
                   root=os.path.join(get_home_dir(), 'models'),
                   pretrained_allow_missing=False, ignore_extra=False,
                   hparam_allow_override=False, **kwargs):
    """Any BERT pretrained model.

    Parameters
    ----------
    model_name : str or None, default None
        Options include 'bert_24_1024_16' and 'bert_12_768_12'.
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        The supported datasets for model_name of either bert_24_1024_16 and
        bert_12_768_12 are 'book_corpus_wiki_en_cased',
        'book_corpus_wiki_en_uncased'.
        For model_name bert_12_768_12 'wiki_cn_cased',
        'wiki_multilingual_uncased', 'wiki_multilingual_cased',
        'scibert_scivocab_uncased', 'scibert_scivocab_cased',
        'scibert_basevocab_uncased','scibert_basevocab_cased',
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed',
        'clinicalbert',
        'kobert_news_wiki_ko_cased'
        are additionally supported.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    use_pooler : bool, default True
        Whether to include the pooler which converts the encoded sequence tensor of shape
        (batch_size, seq_length, units) to a tensor of shape (batch_size, units)
        for for segment level classification task.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
        Note that
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed',
        'clinicalbert'
        do not include these parameters.
    use_classifier : bool, default True
        Whether to include the classifier for next sentence classification.
        Note that
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed'
        do not include these parameters.
    output_attention : bool, default False
        Whether to include attention weights of each encoding cell to the output.
    output_all_encodings : bool, default False
        Whether to output encodings of all encoder cells.
    pretrained_allow_missing : bool, default False
        Whether to ignore if any parameters for the BERTModel are missing in
        the pretrained weights for model.
        Some BERTModels for example do not provide decoder or classifier
        weights. In that case it is still possible to construct a BERTModel
        with use_decoder=True and/or use_classifier=True, but the respective
        parameters will be missing from the pretrained file.
        If pretrained_allow_missing=True, this will be ignored and the
        parameters will be left uninitialized. Otherwise AssertionError is
        raised.
    ignore_extra : bool, default False
        Whether to silently ignore parameters from the file that are not
        present in this Block.
    hparam_allow_override : bool, default False
        If set to True, pre-defined hyper-parameters of the model
        (e.g. the number of layers, hidden units) can be overriden.

    Returns
    -------
    (BERTModel, gluonnlp.vocab.BERTVocab)
    """
    predefined_args = bert_hparams[model_name].copy()
    if not hparam_allow_override:
        mutable_args = ['use_residual', 'dropout', 'word_embed']
        mutable_args = frozenset(mutable_args)
        assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
            'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    # encoder
    encoder = BERTEncoder(num_layers=predefined_args['num_layers'],
                          units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'],
                          dropout=predefined_args['dropout'],
                          output_attention=output_attention,
                          output_all_encodings=output_all_encodings,
                          activation=predefined_args.get('activation', 'gelu'),
                          layer_norm_eps=predefined_args.get('layer_norm_eps', 1e-12))

    from ..vocab import BERTVocab  # pylint: disable=import-outside-toplevel
    bert_vocab = _load_vocab(dataset_name, vocab, root, cls=BERTVocab)
    # BERT
    net = BERTModel(encoder, len(bert_vocab),
                    token_type_vocab_size=predefined_args['token_type_vocab_size'],
                    units=predefined_args['units'],
                    embed_size=predefined_args['embed_size'],
                    word_embed=predefined_args['word_embed'],
                    use_pooler=use_pooler, use_decoder=use_decoder,
                    use_classifier=use_classifier,
                    use_token_type_embed=use_token_type_embed)
    if pretrained:
        ignore_extra = ignore_extra or not (use_pooler and use_decoder and use_classifier)
        _load_pretrained_params(net, model_name, dataset_name, root, ctx, ignore_extra=ignore_extra,
                                allow_missing=pretrained_allow_missing)
    return net, bert_vocab
