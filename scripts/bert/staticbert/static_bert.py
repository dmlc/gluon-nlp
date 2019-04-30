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
"""Static BERT models."""

__all__ = ['StaticBERTModel', 'StaticBERTEncoder',
           'get_model', 'bert_12_768_12', 'bert_24_1024_16', 'get_static_bert_model']

import os
import math
import warnings

from mxnet.gluon import Block, HybridBlock
from mxnet.gluon import nn
import mxnet as mx
from gluonnlp.model.block import GELU
from gluonnlp.model.bert import BERTLayerNorm, BERTEncoderCell, _load_vocab, \
    _load_pretrained_params, bert_hparams
from gluonnlp.model.transformer import TransformerEncoderCell, _get_layer_norm, \
    _position_encoding_init
from gluonnlp.vocab import BERTVocab
from gluonnlp.base import get_home_dir


###############################################################################
#                              COMPONENTS                                     #
###############################################################################


class StaticBaseTransformerEncoder(HybridBlock):
    """Base Structure of the Static Transformer Encoder.

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
    output_attention: bool, default False
        Whether to output the attention weights
    output_all_encodings: bool, default False
        Whether to output encodings of all encoder's cells, or only the last one
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
    input_size : int, default None
        Represents the embedding size of the input.
    seq_length : int, default None
        Stands for the sequence length of the input.
    """

    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False, output_all_encodings=False,
                 weight_initializer=None, bias_initializer='zeros',
                 positional_weight='sinusoidal', use_bert_encoder=False,
                 use_layer_norm_before_dropout=False, scale_embed=True, input_size=None,
                 seq_length=None, prefix=None, params=None):
        super(StaticBaseTransformerEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, \
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
                .format(units, num_heads)
        self._num_layers = num_layers
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        self._use_layer_norm_before_dropout = use_layer_norm_before_dropout
        self._scale_embed = scale_embed
        self._input_size = input_size
        self._seq_length = seq_length
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = _get_layer_norm(use_bert_encoder, units)
            self.position_weight = self._get_positional(positional_weight, max_length, units,
                                                        weight_initializer)
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                cell = self._get_encoder_cell(use_bert_encoder, units, hidden_size, num_heads,
                                              attention_cell, weight_initializer, bias_initializer,
                                              dropout, use_residual, scaled, output_attention, i)
                self.transformer_cells.add(cell)

    def _get_positional(self, weight_type, max_length, units, initializer):
        if weight_type == 'sinusoidal':
            encoding = _position_encoding_init(max_length, units)
            position_weight = self.params.get_constant('const', encoding)
        elif weight_type == 'learned':
            position_weight = self.params.get('position_weight', shape=(max_length, units),
                                              init=initializer)
        else:
            raise ValueError(
                'Unexpected value for argument position_weight: %s' % (position_weight))
        return position_weight

    def _get_encoder_cell(self, use_bert, units, hidden_size, num_heads, attention_cell,
                          weight_initializer, bias_initializer, dropout, use_residual,
                          scaled, output_attention, i):
        cell = BERTEncoderCell if use_bert else TransformerEncoderCell
        return cell(units=units, hidden_size=hidden_size,
                    num_heads=num_heads, attention_cell=attention_cell,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer,
                    dropout=dropout, use_residual=use_residual,
                    scaled=scaled, output_attention=output_attention,
                    prefix='transformer%d_' % i)

    def hybrid_forward(self, F, inputs, states=None,
                       valid_length=None, steps=None,
                       position_weight=None):
        # pylint: disable=arguments-differ
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
        length = self._seq_length
        C_in = self._input_size
        if valid_length is not None:
            arange = F.arange(length)
            mask = F.broadcast_lesser(
                arange.reshape((1, -1)),
                valid_length.reshape((-1, 1)))
            mask = F.broadcast_axes(F.expand_dims(mask, axis=1), axis=1, size=length)
            if states is None:
                states = [mask]
            else:
                states.append(mask)
        if self._scale_embed:
            inputs = inputs * math.sqrt(C_in)
        steps = F.arange(length)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        if states is not None:
            steps = states[-1]
            # Positional Encoding
            positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
            inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=0))
        if self._dropout:
            if self._use_layer_norm_before_dropout:
                inputs = self.layer_norm(inputs)
                inputs = self.dropout_layer(inputs)
            else:
                inputs = self.dropout_layer(inputs)
                inputs = self.layer_norm(inputs)
        else:
            inputs = self.layer_norm(inputs)
        outputs = inputs
        if valid_length is not None:
            mask = states[-2]
        else:
            mask = None

        all_encodings_outputs = []
        additional_outputs = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, mask)
            inputs = outputs
            if self._output_all_encodings:
                if valid_length is not None:
                    outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                             use_sequence_length=True, axis=1)
                all_encodings_outputs.append(outputs)

            if self._output_attention:
                additional_outputs.append(attention_weights)

        if valid_length is not None:
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=1)

        if self._output_all_encodings:
            return all_encodings_outputs, additional_outputs
        else:
            return outputs, additional_outputs


class StaticBERTEncoder(StaticBaseTransformerEncoder):
    """Structure of the Static BERT Encoder.

    Different from the original encoder for transformer,
    `StaticBERTEncoder` uses learnable positional embedding, `BERTPositionwiseFFN`
    and `BERTLayerNorm`.

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
    input_size : int, default None
        Represents the embedding size of the input.
    seq_length : int, default None
        Stands for the sequence length of the input.

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
                 use_residual=True, output_attention=False, output_all_encodings=False,
                 weight_initializer=None, bias_initializer='zeros', input_size=None,
                 seq_length=None, prefix=None, params=None):
        super(StaticBERTEncoder, self).__init__(attention_cell=attention_cell,
                                                num_layers=num_layers, units=units,
                                                hidden_size=hidden_size, max_length=max_length,
                                                num_heads=num_heads, scaled=scaled, dropout=dropout,
                                                use_residual=use_residual,
                                                output_attention=output_attention,
                                                output_all_encodings=output_all_encodings,
                                                weight_initializer=weight_initializer,
                                                bias_initializer=bias_initializer,
                                                prefix=prefix, params=params,
                                                # extra configurations for BERT
                                                positional_weight='learned',
                                                use_bert_encoder=True,
                                                use_layer_norm_before_dropout=False,
                                                scale_embed=False,
                                                input_size=input_size,
                                                seq_length=seq_length)


###############################################################################
#                                FULL MODEL                                   #
###############################################################################

class StaticBERTModel(HybridBlock):
    """Static Model for BERT (Bidirectional Encoder Representations from Transformers).

    Parameters
    ----------
    encoder : StaticBERTEncoder
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
        See document of `mx.gluon.HybridBlock`.
    params : ParameterDict or None
        See document of `mx.gluon.HybridBlock`.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **token_types**: input token type tensor, shape (batch_size, seq_length).
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
            Returned only if StaticBERTEncoder.output_attention is True.
            List of num_layers length of tensors of shape
            (num_masks, num_attention_heads, seq_length, seq_length)
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
        super(StaticBERTModel, self).__init__(prefix=prefix, params=params)
        self._use_decoder = use_decoder
        self._use_classifier = use_classifier
        self._use_pooler = use_pooler
        self._vocab_size = vocab_size
        self.encoder = encoder
        # Construct word embedding
        self.word_embed = self._get_embed(word_embed, vocab_size, embed_size,
                                          embed_initializer, embed_dropout, 'word_embed_')
        # Construct token type embedding
        self.token_type_embed = self._get_embed(token_type_embed, token_type_vocab_size,
                                                embed_size, embed_initializer, embed_dropout,
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
        """ Construct a decoder for the masked language model task """
        with self.name_scope():
            classifier = nn.Dense(2, prefix=prefix)
        return classifier

    def _get_decoder(self, units, vocab_size, embed, prefix):
        """ Construct a decoder for the masked language model task """
        with self.name_scope():
            decoder = nn.HybridSequential(prefix=prefix)
            decoder.add(nn.Dense(units, flatten=False))
            decoder.add(GELU())
            decoder.add(BERTLayerNorm(in_channels=units))
            decoder.add(nn.Dense(vocab_size, flatten=False, params=embed.collect_params()))
        assert decoder[3].weight == list(embed.collect_params().values())[0], \
            'The weights of word embedding are not tied with those of decoder'
        return decoder

    def _get_embed(self, embed, vocab_size, embed_size, initializer, dropout, prefix):
        """ Construct an embedding block. """
        if embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "word_embed" or ' \
                                           'token_type_embed is not given.'
            with self.name_scope():
                embed = nn.HybridSequential(prefix=prefix)
                with embed.name_scope():
                    embed.add(nn.Embedding(input_dim=vocab_size, output_dim=embed_size,
                                           weight_initializer=initializer))
                    if dropout:
                        embed.add(nn.Dropout(rate=dropout))
        assert isinstance(embed, Block)
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

    def hybrid_forward(self, F, inputs, token_types, valid_length=None, masked_positions=None):
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a static (hybridized) BERT model.
        """
        outputs = []
        seq_out, attention_out = self._encode_sequence(F, inputs, token_types, valid_length)
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
            decoder_out = self._decode(output, masked_positions)
            outputs.append(decoder_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def _encode_sequence(self, F, inputs, token_types, valid_length=None):
        # pylint: disable=unused-argument
        """Generate the representation given the input sequences.

        This is used for pre-training or fine-tuning a static (hybridized) BERT model.
        """
        # embedding
        word_embedding = self.word_embed(inputs)
        type_embedding = self.token_type_embed(token_types)
        embedding = word_embedding + type_embedding
        # encoding
        outputs, additional_outputs = self.encoder(embedding, None, valid_length)
        return outputs, additional_outputs

    def _apply_pooling(self, sequence):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a BERT model.
        """
        outputs = sequence[:, 0, :]
        return self.pooler(outputs)

    def _decode(self, sequence, masked_positions):
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
        batch_size = sequence.shape[0]
        num_masked_positions = masked_positions.shape[1]
        ctx = masked_positions.context
        dtype = masked_positions.dtype
        # batch_idx = [0,0,0,1,1,1,2,2,2...]
        # masked_positions = [1,2,4,0,3,4,2,3,5...]
        batch_idx = mx.nd.arange(0, batch_size, repeat=num_masked_positions, dtype=dtype, ctx=ctx)
        batch_idx = batch_idx.reshape((1, -1))
        masked_positions = masked_positions.reshape((1, -1))
        position_idx = mx.nd.Concat(batch_idx, masked_positions, dim=0)
        encoded = mx.nd.gather_nd(sequence, position_idx)
        encoded = encoded.reshape((batch_size, num_masked_positions, sequence.shape[-1]))
        decoded = self.decoder(encoded)
        return decoded


###############################################################################
#                               GET MODEL                                     #
###############################################################################

def get_model(name, dataset_name='wikitext-2', **kwargs):
    """Returns a pre-defined model by name.

    Parameters
    ----------
    name : str
        Name of the model.
    dataset_name : str or None, default 'wikitext-2'.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
        None Vocabulary object is required with the ELMo model.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab, (optional) gluonnlp.Vocab
    """
    models = {'bert_12_768_12': bert_12_768_12,
              'bert_24_1024_16': bert_24_1024_16}
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s' % (
                name, '\n\t'.join(sorted(models.keys()))))
    kwargs['dataset_name'] = dataset_name
    return models[name](**kwargs)


def bert_12_768_12(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                   root=os.path.join(get_home_dir(), 'models'), use_pooler=True,
                   use_decoder=True, use_classifier=True, input_size=None, seq_length=None,
                   **kwargs):
    """Static BERT BASE model.

    The number of layers (L) is 12, number of units (H) is 768, and the
    number of self-attention heads (A) is 12.

    Parameters
    ----------
    dataset_name : str or None, default None
        Options include 'book_corpus_wiki_en_cased', 'book_corpus_wiki_en_uncased',
        'wiki_cn_cased', 'wiki_multilingual_uncased' and 'wiki_multilingual_cased'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
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
    input_size : int, default None
        Represents the embedding size of the input.
    seq_length : int, default None
        Stands for the sequence length of the input.

    Returns
    -------
    StaticBERTModel, gluonnlp.vocab.BERTVocab
    """
    return get_static_bert_model(model_name='bert_12_768_12', vocab=vocab,
                                 dataset_name=dataset_name, pretrained=pretrained, ctx=ctx,
                                 use_pooler=use_pooler, use_decoder=use_decoder,
                                 use_classifier=use_classifier, root=root, input_size=input_size,
                                 seq_length=seq_length, **kwargs)


def bert_24_1024_16(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                    use_pooler=True, use_decoder=True, use_classifier=True,
                    root=os.path.join(get_home_dir(), 'models'), input_size=None, seq_length=None,
                    **kwargs):
    """Static BERT LARGE model.

    The number of layers (L) is 24, number of units (H) is 1024, and the
    number of self-attention heads (A) is 16.

    Parameters
    ----------
    dataset_name : str or None, default None
        Options include 'book_corpus_wiki_en_uncased' and 'book_corpus_wiki_en_cased'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
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
    input_size : int, default None
        Represents the embedding size of the input.
    seq_length : int, default None
        Stands for the sequence length of the input.

    Returns
    -------
    StaticBERTModel, gluonnlp.vocab.BERTVocab
    """
    return get_static_bert_model(model_name='bert_24_1024_16', vocab=vocab,
                                 dataset_name=dataset_name, pretrained=pretrained,
                                 ctx=ctx, use_pooler=use_pooler,
                                 use_decoder=use_decoder, use_classifier=use_classifier,
                                 root=root, input_size=input_size, seq_length=seq_length, **kwargs)


def get_static_bert_model(model_name=None, dataset_name=None, vocab=None,
                          pretrained=True, ctx=mx.cpu(),
                          use_pooler=True, use_decoder=True, use_classifier=True,
                          output_attention=False, output_all_encodings=False,
                          root=os.path.join(get_home_dir(), 'models'), input_size=None,
                          seq_length=None, **kwargs):
    """Any Static BERT pretrained model.

    Parameters
    ----------
    model_name : str or None, default None
        Options include 'bert_24_1024_16' and 'bert_12_768_12'.
    dataset_name : str or None, default None
        Options include 'book_corpus_wiki_en_cased', 'book_corpus_wiki_en_uncased'
        for both bert_24_1024_16 and bert_12_768_12.
        'wiki_cn_cased', 'wiki_multilingual_uncased' and 'wiki_multilingual_cased'
        for bert_12_768_12 only.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
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
    output_attention : bool, default False
        Whether to include attention weights of each encoding cell to the output.
    output_all_encodings : bool, default False
        Whether to output encodings of all encoder cells.
    input_size : int, default None
        Represents the embedding size of the input.
    seq_length : int, default None
        Stands for the sequence length of the input.

    Returns
    -------
    StaticBERTModel, gluonnlp.vocab.BERTVocab
    """
    predefined_args = bert_hparams[model_name]
    mutable_args = ['use_residual', 'dropout', 'embed_dropout', 'word_embed']
    mutable_args = frozenset(mutable_args)
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
        'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    # encoder
    encoder = StaticBERTEncoder(attention_cell=predefined_args['attention_cell'],
                                num_layers=predefined_args['num_layers'],
                                units=predefined_args['units'],
                                hidden_size=predefined_args['hidden_size'],
                                max_length=predefined_args['max_length'],
                                num_heads=predefined_args['num_heads'],
                                scaled=predefined_args['scaled'],
                                dropout=predefined_args['dropout'],
                                output_attention=output_attention,
                                output_all_encodings=output_all_encodings,
                                use_residual=predefined_args['use_residual'],
                                input_size=input_size,
                                seq_length=seq_length)
    if dataset_name in ['wiki_cn', 'wiki_multilingual']:
        warnings.warn('wiki_cn/wiki_multilingual will be deprecated.'
                      ' Please use wiki_cn_cased/wiki_multilingual_uncased instead.')
    bert_vocab = _load_vocab(dataset_name, vocab, root, cls=BERTVocab)
    # BERT
    net = StaticBERTModel(encoder, len(bert_vocab),
                          token_type_vocab_size=predefined_args['token_type_vocab_size'],
                          units=predefined_args['units'],
                          embed_size=predefined_args['embed_size'],
                          embed_dropout=predefined_args['embed_dropout'],
                          word_embed=predefined_args['word_embed'],
                          use_pooler=use_pooler, use_decoder=use_decoder,
                          use_classifier=use_classifier)
    if pretrained:
        ignore_extra = not (use_pooler and use_decoder and use_classifier)
        _load_pretrained_params(net, model_name, dataset_name, root, ctx,
                                ignore_extra=ignore_extra)
    return net, bert_vocab
