# coding: utf-8

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
"""Language models."""

__all__ = ['bert_lm_12_768_12_300_1150', 'bert_lm_12_768_12_400_2500',
           'bert_lm_24_1024_16_300_1150', 'bert_lm_24_1024_16_400_2500']

import os
import math

import mxnet as mx

from mxnet import nd
from mxnet.gluon.model_zoo import model_store

from gluonnlp.model import BERTModel
from gluonnlp.model.transformer import BaseTransformerEncoder
from gluonnlp.model.utils import _load_vocab, _load_pretrained_params

try:
    from . import train
except ImportError:
    import train


class BaseTransformerMaskedEncoder(BaseTransformerEncoder):
    """Base Structure of the Transformer Masked Encoder.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
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
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities.
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    positional_weight: str, default 'sinusoidal'
        Type of positional embedding. Can be 'sinusoidal', 'learned'.
        If set to 'sinusoidal', the embedding is initialized as sinusoidal values and keep constant.
    use_bert_encoder : bool, default False
        Whether to use BERTEncoderCell and BERTLayerNorm. Set to True for pre-trained BERT model
    use_layer_norm_before_dropout: bool, default False
        Before passing embeddings to attention cells, whether to perform `layernorm -> dropout` or
        `dropout -> layernorm`. Set to True for pre-trained BERT models.
    scale_embed : bool, default True
        Scale the input embeddings by sqrt(embed_size). Set to False for pre-trained BERT models.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """

    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 positional_weight='sinusoidal', use_bert_encoder=False,
                 use_layer_norm_before_dropout=False, scale_embed=True,
                 prefix=None, params=None):
        super(BaseTransformerMaskedEncoder, self).__init__(attention_cell=attention_cell,
                                                           num_layers=num_layers,
                                                           units=units, hidden_size=hidden_size,
                                                           max_length=max_length,
                                                           num_heads=num_heads, scaled=scaled,
                                                           dropout=dropout,
                                                           use_residual=use_residual,
                                                           output_attention=output_attention,
                                                           weight_initializer=weight_initializer,
                                                           bias_initializer=bias_initializer,
                                                           positional_weight=positional_weight,
                                                           use_bert_encoder=use_bert_encoder,
                                                           use_layer_norm_before_dropout
                                                           =use_layer_norm_before_dropout,
                                                           scale_embed=scale_embed,
                                                           prefix=prefix, params=params)

    def forward(self, inputs, states=None, valid_length=None,
                steps=None):  # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray, Shape(batch_size, length, C_in)
        states : list of NDArray
        valid_length : NDArray
        steps : NDArray
            Stores value [0, 1, ..., length].
            It is used for lookup in positional encoding matrix

        Returns
        -------
        outputs : NDArray
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        batch_size = inputs.shape[0]
        length = inputs.shape[1]
        if valid_length is None:
            valid_length = mx.nd.array([length] * batch_size, ctx=inputs.context)
        length_array = mx.nd.arange(length, ctx=inputs.context)
        mask = mx.nd.broadcast_lesser_equal(
            length_array.reshape((1, -1)),
            length_array.reshape((-1, 1)))
        mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0, size=batch_size)
        if states is None:
            states = [mask]
        else:
            states.append(mask)
        if self._scale_embed:
            inputs = inputs * math.sqrt(inputs.shape[-1])
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        # pylint: disable=E1003
        step_output, additional_outputs = \
            super(BaseTransformerEncoder, self).forward(inputs, states, valid_length)
        return step_output, additional_outputs

    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None):
        # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        states : list of NDArray or Symbol
        valid_length : NDArray or Symbol
        position_weight : NDArray or Symbol

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        if states is not None:
            steps = states[-1]
            # Positional Encoding
            positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
            inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=0))
        if self._use_layer_norm_before_dropout:
            inputs = self.layer_norm(inputs)
            inputs = self.dropout_layer(inputs)
        else:
            inputs = self.dropout_layer(inputs)
            inputs = self.layer_norm(inputs)
        outputs = inputs
        mask = states[-2]
        additional_outputs = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, mask)
            inputs = outputs
            if self._output_attention:
                additional_outputs.append(attention_weights)
        if valid_length is not None:
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=1)
        return outputs, additional_outputs


class BERTMaskedEncoder(BaseTransformerMaskedEncoder):
    """Structure of the BERT Masked Encoder.

    Different from the original encoder for transformer and `BERTEncoder` ,
    `BERTMaskedEncoder` uses masks to prevent leftward information flow.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
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
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities.
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None.
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.

    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in)
        - **states** : list of tensors for initial states and masks.
        - **valid_length** : valid lengths of each sequence. Usually used when part of sequence
            has been padded. Shape is (batch_size, )

    Outputs:
        - **outputs** : the output of the encoder. Shape is (batch_size, length, C_out)
        - **additional_outputs** : list of tensors.
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, num_heads, length, mem_length)
    """

    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BERTMaskedEncoder, self).__init__(attention_cell=attention_cell,
                                                num_layers=num_layers, units=units,
                                                hidden_size=hidden_size, max_length=max_length,
                                                num_heads=num_heads, scaled=scaled, dropout=dropout,
                                                use_residual=use_residual,
                                                output_attention=output_attention,
                                                weight_initializer=weight_initializer,
                                                bias_initializer=bias_initializer,
                                                prefix=prefix, params=params,
                                                # extra configurations for BERT
                                                positional_weight='learned',
                                                use_bert_encoder=True,
                                                use_layer_norm_before_dropout=False,
                                                scale_embed=False)


class BERTMaskedModel(BERTModel):
    """Model for BERT (Bidirectional Encoder Representations from Transformers)
    used in Language Model.

    Parameters
    ----------
    encoder : BERTEncoder
        Bidirectional encoder that encodes the input sentence.
    vocab_size : int or None, default None
        The size of the vocabulary.
    token_type_vocab_size : int or None, default None
        The vocabulary size of token types.
    units : int or None, default None
        Number of units for the final pooler layer.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the word and token type
        embeddings if word_embed and token_type_embed are None.
    embed_dropout : float, default 0.0
        Dropout rate of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    embed_initializer : Initializer, default None
        Initializer of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    word_embed : Block or None, default None
        The word embedding. If set to None, word_embed will be constructed using embed_size and
        embed_dropout.
    token_type_embed : Block or None, default None
        The token type embedding. If set to None and the token_type_embed will be constructed using
        embed_size and embed_dropout.
    use_pooler : bool, default True
        Whether to include the pooler which converts the encoded sequence tensor of shape
        (batch_size, seq_length, units) to a tensor of shape (batch_size, units)
        for segment level classification task.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
    use_classifier : bool, default True
        Whether to include the classifier for next sentence classification.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **token_types**: input token type tensor, shape (batch_size, seq_length).
            If the inputs contain two sequences, then the token type of the first
            sequence differs from that of the second one.
        - **valid_length**: optional tensor of input sequence valid lengths, shape (batch_size,)
        - **masked_positions**: optional tensor of position of tokens for masked LM decoding,
            shape (batch_size, num_masked_positions).

    Outputs:
        - **sequence_outputs**: output tensor of sequence encodings.
            Shape (batch_size, seq_length, units).
        - **pooled_output**: output tensor of pooled representation of the first tokens.
            Returned only if use_pooler is True. Shape (batch_size, units)
        - **next_sentence_classifier_output**: output tensor of next sentence classification.
            Returned only if use_classifier is True. Shape (batch_size, 2)
        - **masked_lm_outputs**: output tensor of sequence decoding for masked language model
            prediction. Returned only if use_decoder True.
            Shape (batch_size, num_masked_positions, vocab_size)
    """

    def __init__(self, encoder, vocab_size=None, token_type_vocab_size=None, units=None,
                 embed_size=None, embed_dropout=0.0, embed_initializer=None,
                 word_embed=None, token_type_embed=None, use_pooler=True, use_decoder=True,
                 use_classifier=True, prefix=None, params=None):
        super(BERTMaskedModel, self).__init__(encoder=encoder, vocab_size=vocab_size,
                                              token_type_vocab_size=token_type_vocab_size,
                                              units=units,
                                              embed_size=embed_size, embed_dropout=embed_dropout,
                                              embed_initializer=embed_initializer,
                                              word_embed=word_embed,
                                              token_type_embed=token_type_embed,
                                              use_pooler=use_pooler, use_decoder=use_decoder,
                                              use_classifier=use_classifier, prefix=prefix,
                                              params=params)

    def __call__(self, inputs, token_types, valid_length=None, masked_positions=None):
        return super(BERTMaskedModel, self).__call__(inputs, token_types, valid_length,
                                                     masked_positions)


###############################################################################
#                               BERT based LANGUAGE MODEL                     #
###############################################################################


class BERTRNN(train.BERTRNN):
    """BERT based language model. Paper would be available soon.

    Parameters
    ----------
    embedding : BERTMaskedModel
        The BERT structure to use.
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    hidden_size_last : int
        Number of last hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    dropout : float
        Dropout rate to use for encoder output.
    weight_drop : float
        Dropout rate to use on encoder h2h weights.
    drop_h : float
        Dropout rate to on the output of intermediate layers of encoder.
    drop_i : float
        Dropout rate to on the output of embedding.
    drop_e : float
        Dropout rate to use on the embedding layer.
    drop_l : float
        Dropout rate to use on the latent layer.
    num_experts : int
        Number of experts in mixture of softmax.
    upperbound_fixed_layer : int
        Number of layers in BERT with the parameters fixed.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **begin_state**: initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`
        - **token_types**: input token type tensor, shape (batch_size, seq_length).
            If the inputs contain two sequences, then the token type of the first
            sequence differs from that of the second one.
        - **valid_length**: optional tensor of input sequence valid lengths, shape (batch_size,)
        - **masked_positions**: optional tensor of position of tokens for masked LM decoding,
            shape (batch_size, num_masked_positions).

    Outputs:
        - **out**: output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        - **out_states**: output recurrent state tensor with length equals to num_layers.
            the state with shape `(1, batch_size, num_hidden)`
    """

    def __init__(self, embedding, mode, vocab_size,
                 embed_size=300, hidden_size=1150,
                 hidden_size_last=650,
                 num_layers=3, tie_weights=True,
                 dropout=0.4, weight_drop=0.5, drop_h=0.2,
                 drop_i=0.55, drop_e=0.1, drop_l=0.29,
                 num_experts=15, upperbound_fixed_layer=22, **kwargs):
        super(BERTRNN, self).__init__(embedding=embedding, mode=mode, vocab_size=vocab_size,
                                      embed_size=embed_size, hidden_size=hidden_size,
                                      hidden_size_last=hidden_size_last,
                                      num_layers=num_layers, tie_weights=tie_weights,
                                      dropout=dropout, weight_drop=weight_drop, drop_h=drop_h,
                                      drop_i=drop_i, drop_e=drop_e, drop_l=drop_l,
                                      num_experts=num_experts,
                                      upperbound_fixed_layer=upperbound_fixed_layer, **kwargs)

    def __call__(self, inputs, begin_state=None, token_types=None, valid_length=None,
                 masked_positions=None):
        return super(BERTRNN, self).__call__(inputs, begin_state, token_types, valid_length,
                                             masked_positions)

    def forward(self, inputs, begin_state=None, token_types=None, valid_length=None,
                masked_positions=None):  # pylint: disable=arguments-differ
        """Implement the forward computation that the awd language model and cache model use.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`
        token_types: NDArray
            input token type tensor, shape (batch_size, seq_length).
            If the inputs contain two sequences, then the token type of the first
            sequence differs from that of the second one.
        valid_length: NDArray
            optional tensor of input sequence valid lengths, shape (batch_size,)
        masked_positions: optional tensor of position of tokens for masked LM decoding,
            shape (batch_size, num_masked_positions).

        Returns
        --------
        out: NDArray
            output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers.
            the state with shape `(1, batch_size, num_hidden)`
        """
        batch_size = inputs.shape[1]
        inputs = nd.transpose(inputs, axes=(1, 0))
        if token_types is None:
            token_types = nd.zeros_like(inputs)
        encoded = self.embedding(inputs, token_types=token_types,
                                 valid_length=valid_length, masked_positions=masked_positions)
        encoded = nd.transpose(encoded, axes=(1, 0, 2))
        encoded = nd.Dropout(encoded, p=self._drop_i, axes=(0,))
        if not begin_state:
            begin_state = self.begin_state(batch_size=batch_size)
        out_states = []
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            out_states.append(state)
            if i != len(self.encoder) - 1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
        encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        # use mos
        latent = nd.Dropout(self.latent(encoded), p=self._drop_l, axes=(0,))
        logit = self.decoder(latent.reshape(-1, self._embed_size))

        prior_logit = self.prior(encoded).reshape(-1, self._num_experts)
        prior = nd.softmax(prior_logit, axis=-1)

        prob = nd.softmax(logit.reshape(-1, self._vocab_size), axis=-1)
        prob = prob.reshape(-1, self._num_experts, self._vocab_size)
        prob = (prob * prior.expand_dims(2).broadcast_to(prob.shape)).sum(axis=1)

        out = nd.log(nd.add(prob, 1e-8)).reshape(-1, batch_size, self._vocab_size)

        return out, out_states

###############################################################################
#                               GET MODEL                                     #
###############################################################################


model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('343d438e1270a492e19276b73330a34a51e46869', 'bert_lm_12_768_12_400_2500_wikitext103'),
        ('e584719d1af5eaed84d1bb3b40a1057346be1c3c', 'bert_lm_24_1024_16_400_2500_wikitext103'),
        ('720439248785a170834718f0c213564b02586182', 'bert_lm_12_768_12_300_1150_wikitext2'),
        ('df96b09a24c81d6bebe34b2ca6e55352f8bd1953', 'bert_lm_24_1024_16_300_1150_wikitext2')
    ]})

bert_lm_12_768_12_300_1150_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 12,
    'units': 768,
    'hidden_size': 3072,
    'max_length': 512,
    'num_heads': 12,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 768,
    'embed_dropout': 0.1,
    'token_type_vocab_size': 2,
    'word_embed': None,
    'upperbound_fixed_layer': 9,
    'rnn_embed_size': 300,
    'rnn_hidden_size': 1150,
    'rnn_hidden_size_last': 650,
    'rnn_mode': 'lstm',
    'rnn_num_layers': 3,
    'rnn_tie_weights': False,
    'rnn_dropout': 0.1,
    'rnn_weight_drop': 0,
    'rnn_drop_h': 0.1,
    'rnn_drop_i': 0.1,
    'rnn_drop_e': 0.1,
    'rnn_drop_l': 0.1,
    'rnn_num_experts': 15
}

bert_lm_12_768_12_400_2500_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 12,
    'units': 768,
    'hidden_size': 3072,
    'max_length': 512,
    'num_heads': 12,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 768,
    'embed_dropout': 0.1,
    'token_type_vocab_size': 2,
    'word_embed': None,
    'upperbound_fixed_layer': 9,
    'rnn_embed_size': 400,
    'rnn_hidden_size': 2500,
    'rnn_hidden_size_last': 400,
    'rnn_mode': 'lstm',
    'rnn_num_layers': 4,
    'rnn_tie_weights': False,
    'rnn_dropout': 0.1,
    'rnn_weight_drop': 0,
    'rnn_drop_h': 0.1,
    'rnn_drop_i': 0.1,
    'rnn_drop_e': 0.1,
    'rnn_drop_l': 0.1,
    'rnn_num_experts': 15
}

bert_lm_24_1024_16_300_1150_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 24,
    'units': 1024,
    'hidden_size': 4096,
    'max_length': 512,
    'num_heads': 16,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 1024,
    'embed_dropout': 0.1,
    'token_type_vocab_size': 2,
    'word_embed': None,
    'upperbound_fixed_layer': 22,
    'rnn_embed_size': 300,
    'rnn_hidden_size': 1150,
    'rnn_hidden_size_last': 650,
    'rnn_mode': 'lstm',
    'rnn_num_layers': 3,
    'rnn_tie_weights': False,
    'rnn_dropout': 0.1,
    'rnn_weight_drop': 0,
    'rnn_drop_h': 0.1,
    'rnn_drop_i': 0.1,
    'rnn_drop_e': 0.1,
    'rnn_drop_l': 0.1,
    'rnn_num_experts': 15
}

bert_lm_24_1024_16_400_2500_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 24,
    'units': 1024,
    'hidden_size': 4096,
    'max_length': 512,
    'num_heads': 16,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 1024,
    'embed_dropout': 0.1,
    'token_type_vocab_size': 2,
    'word_embed': None,
    'upperbound_fixed_layer': 9,
    'rnn_embed_size': 400,
    'rnn_hidden_size': 2500,
    'rnn_hidden_size_last': 400,
    'rnn_mode': 'lstm',
    'rnn_num_layers': 4,
    'rnn_tie_weights': False,
    'rnn_dropout': 0.1,
    'rnn_weight_drop': 0,
    'rnn_drop_h': 0.1,
    'rnn_drop_i': 0.1,
    'rnn_drop_e': 0.1,
    'rnn_drop_l': 0.1,
    'rnn_num_experts': 15
}

# BERT based language model hyper-parameters
bert_lm_hparams = {
    'bert_lm_12_768_12_300_1150': bert_lm_12_768_12_300_1150_hparams,
    'bert_lm_12_768_12_400_2500': bert_lm_12_768_12_400_2500_hparams,
    'bert_lm_24_1024_16_300_1150': bert_lm_24_1024_16_300_1150_hparams,
    'bert_lm_24_1024_16_400_2500': bert_lm_24_1024_16_400_2500_hparams,
}

# The vocabularies used for BERT based language model hyper-parameters
bert_vocabs = {
    'bert_lm_12_768_12_300_1150': 'book_corpus_wiki_en_uncased',
    'bert_lm_12_768_12_400_2500': 'book_corpus_wiki_en_uncased',
    'bert_lm_24_1024_16_300_1150': 'book_corpus_wiki_en_uncased',
    'bert_lm_24_1024_16_400_2500': 'book_corpus_wiki_en_uncased',
}


def bert_lm_12_768_12_300_1150(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                               root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """BERT BASE based pretrained language model for Wikitext-2.

    The number of layers (L) is 12, number of units (H) is 768, and the
    number of self-attention heads (A) is 12, the hidden size of RNN is 1150.

    Parameters
    ----------
    dataset_name : str or None, default None
        Options include 'wikitext2'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.


    Returns
    -------
    BERTRNN, gluonnlp.vocab.BERTVocab
    """
    return _bert_lm_model(model_name='bert_lm_12_768_12_300_1150', vocab=vocab,
                          dataset_name=dataset_name, pretrained=pretrained, ctx=ctx,
                          root=root, **kwargs)


def bert_lm_12_768_12_400_2500(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                               root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """BERT BASE based pretrained language model for Wikitext-103.

    The number of layers (L) is 12, number of units (H) is 768, and the
    number of self-attention heads (A) is 12, the hidden size of RNN is 2500.

    Parameters
    ----------
    dataset_name : str or None, default None
        Options include 'wikitext103'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.


    Returns
    -------
    BERTRNN, gluonnlp.vocab.BERTVocab
    """
    return _bert_lm_model(model_name='bert_lm_12_768_12_400_2500', vocab=vocab,
                          dataset_name=dataset_name, pretrained=pretrained, ctx=ctx,
                          root=root, **kwargs)


def bert_lm_24_1024_16_300_1150(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """BERT LARGE based pretrained language model for Wikitext-2.

    The number of layers (L) is 24, number of units (H) is 1024, and the
    number of self-attention heads (A) is 16, the hidden size of RNN is 1150.]

    Parameters
    ----------
    dataset_name : str or None, default None
        Options include 'wikitext2'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.


    Returns
    -------
    BERTRNN, gluonnlp.vocab.BERTVocab
    """
    return _bert_lm_model(model_name='bert_lm_24_1024_16_300_1150', vocab=vocab,
                          dataset_name=dataset_name, pretrained=pretrained,
                          ctx=ctx,
                          root=root, **kwargs)


def bert_lm_24_1024_16_400_2500(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """BERT LARGE based pretrained language model for Wikitext-103.

    The number of layers (L) is 24, number of units (H) is 1024, and the
    number of self-attention heads (A) is 16, the hidden size of RNN is 2500.

    Parameters
    ----------
    dataset_name : str or None, default None
        Options include 'wikitext103'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.


    Returns
    -------
    BERTRNN, gluonnlp.vocab.BERTVocab
    """
    return _bert_lm_model(model_name='bert_lm_24_1024_16_400_2500', vocab=vocab,
                          dataset_name=dataset_name, pretrained=pretrained,
                          ctx=ctx,
                          root=root, **kwargs)


def _bert_lm_model(model_name=None, dataset_name=None, vocab=None,
                   pretrained=True, ctx=mx.cpu(),
                   root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """BERT based pretrained language model.

    Returns
    -------
    BERTRNN, gluonnlp.vocab.BERTVocab
    """
    predefined_args = bert_lm_hparams[model_name]
    mutable_args = ['use_residual', 'dropout', 'embed_dropout', 'word_embed',
                    'rnn_dropout', 'rnn_weight_drop', 'rnn_drop_h', 'rnn_drop_i',
                    'rnn_drop_e', 'rnn_drop_l']
    mutable_args = frozenset(mutable_args)
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
        'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    # encoder
    encoder = BERTMaskedEncoder(attention_cell=predefined_args['attention_cell'],
                                num_layers=predefined_args['num_layers'],
                                units=predefined_args['units'],
                                hidden_size=predefined_args['hidden_size'],
                                max_length=predefined_args['max_length'],
                                num_heads=predefined_args['num_heads'],
                                scaled=predefined_args['scaled'],
                                dropout=predefined_args['dropout'],
                                use_residual=predefined_args['use_residual'])
    # bert_vocab
    from gluonnlp.vocab.bert import BERTVocab
    bert_vocab = _load_vocab(bert_vocabs[model_name], vocab, root, cls=BERTVocab)
    # BERT
    bert = BERTMaskedModel(encoder, len(bert_vocab),
                           token_type_vocab_size=predefined_args['token_type_vocab_size'],
                           units=predefined_args['units'],
                           embed_size=predefined_args['embed_size'],
                           embed_dropout=predefined_args['embed_dropout'],
                           word_embed=predefined_args['word_embed'],
                           use_pooler=False, use_decoder=False,
                           use_classifier=False)

    # BERT LM
    net = BERTRNN(embedding=bert, mode=predefined_args['rnn_mode'], vocab_size=len(bert_vocab),
                  embed_size=predefined_args['rnn_embed_size'],
                  hidden_size=predefined_args['rnn_hidden_size'],
                  hidden_size_last=predefined_args['rnn_hidden_size_last'],
                  num_layers=predefined_args['rnn_num_layers'],
                  tie_weights=predefined_args['rnn_tie_weights'],
                  dropout=predefined_args['rnn_dropout'],
                  weight_drop=predefined_args['rnn_weight_drop'],
                  drop_h=predefined_args['rnn_drop_h'],
                  drop_i=predefined_args['rnn_drop_i'],
                  drop_e=predefined_args['rnn_drop_e'],
                  drop_l=predefined_args['rnn_drop_l'],
                  num_experts=predefined_args['rnn_num_experts'],
                  upperbound_fixed_layer=predefined_args['upperbound_fixed_layer'], **kwargs)

    if pretrained:
        _load_pretrained_params(net, model_name, dataset_name, root, ctx,
                                ignore_extra=True)
    return net, bert_vocab
