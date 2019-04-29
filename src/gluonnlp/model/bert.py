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
"""BERT models."""

__all__ = ['BERTModel', 'BERTEncoder', 'BERTEncoderCell', 'BERTPositionwiseFFN',
           'BERTLayerNorm', 'bert_12_768_12', 'bert_24_1024_16', 'get_bert_model']

import os
import warnings
from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import model_store
import mxnet as mx
from .transformer import BasePositionwiseFFN, BaseTransformerEncoderCell, BaseTransformerEncoder
from .block import GELU
from .utils import _load_vocab, _load_pretrained_params
from ..base import get_home_dir


###############################################################################
#                              COMPONENTS                                     #
###############################################################################


class BERTLayerNorm(nn.LayerNorm):
    """BERT style Layer Normalization, where epsilon is added inside the square
    root and set to 1e-12 by default.

    Inputs:
        - **data**: input tensor with arbitrary shape.
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, epsilon=1e-12, in_channels=0, prefix=None, params=None):
        super(BERTLayerNorm, self).__init__(epsilon=epsilon, in_channels=in_channels,
                                            prefix=prefix, params=params)
        self._dtype = None

    def cast(self, dtype):
        self._dtype = dtype
        super(BERTLayerNorm, self).cast('float32')

    def hybrid_forward(self, F, data, gamma, beta):
        """forward computation."""
        # TODO(haibin): LayerNorm does not support fp16 safe reduction. Issue is tracked at:
        # https://github.com/apache/incubator-mxnet/issues/14073
        if self._dtype:
            data = data.astype('float32')
            gamma = gamma.astype('float32')
            beta = beta.astype('float32')
        norm_data = F.LayerNorm(data, gamma=gamma, beta=beta, axis=self._axis, eps=self._epsilon)
        if self._dtype:
            norm_data = norm_data.astype(self._dtype)
        return norm_data


class BERTPositionwiseFFN(BasePositionwiseFFN):
    """Structure of the Positionwise Feed-Forward Neural Network for
    BERT.

    Different from the original positionwise feed forward network
    for transformer, `BERTPositionwiseFFN` uses `GELU` for activation
    and `BERTLayerNorm` for layer normalization.

    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        Number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
        Dropout probability for the output
    use_residual : bool
        Add residual connection between the input and the output
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.

    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in).

    Outputs:
        - **outputs** : output encoding of shape (batch_size, length, C_out).
    """

    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BERTPositionwiseFFN, self).__init__(units=units, hidden_size=hidden_size,
                                                  dropout=dropout, use_residual=use_residual,
                                                  weight_initializer=weight_initializer,
                                                  bias_initializer=bias_initializer,
                                                  prefix=prefix, params=params,
                                                  # extra configurations for BERT
                                                  activation='gelu',
                                                  use_bert_layer_norm=True)


class BERTEncoder(BaseTransformerEncoder):
    """Structure of the BERT Encoder.

    Different from the original encoder for transformer,
    `BERTEncoder` uses learnable positional embedding, `BERTPositionwiseFFN`
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
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BERTEncoder, self).__init__(attention_cell=attention_cell,
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
                                          scale_embed=False)


class BERTEncoderCell(BaseTransformerEncoderCell):
    """Structure of the Transformer Encoder Cell for BERT.

    Different from the original encoder cell for transformer,
    `BERTEncoderCell` adds bias terms for attention and the projection
    on attention output. It also uses `BERTPositionwiseFFN` and
    `BERTLayerNorm`.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
        Number of units for the output
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.

    Inputs:
        - **inputs** : input sequence. Shape (batch_size, length, C_in)
        - **mask** : mask for inputs. Shape (batch_size, length, length)

    Outputs:
        - **outputs**: output tensor of the transformer encoder cell.
            Shape (batch_size, length, C_out)
        - **additional_outputs**: the additional output of all the transformer encoder cell.
    """

    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BERTEncoderCell, self).__init__(attention_cell=attention_cell,
                                              units=units, hidden_size=hidden_size,
                                              num_heads=num_heads, scaled=scaled,
                                              dropout=dropout, use_residual=use_residual,
                                              output_attention=output_attention,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              prefix=prefix, params=params,
                                              # extra configurations for BERT
                                              attention_use_bias=True,
                                              attention_proj_use_bias=True,
                                              use_bert_layer_norm=True,
                                              use_bert_ffn=True)

###############################################################################
#                                FULL MODEL                                   #
###############################################################################

class BERTModel(Block):
    """Generic Model for BERT (Bidirectional Encoder Representations from Transformers).

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
        - **sequence_outputs**: Encoded sequence, which can be either a tensor of the last
            layer of the Encoder, or a list of all sequence encodings of all layers.
            In both cases shape of the tensor(s) is/are (batch_size, seq_length, units).
        - **attention_outputs**: output list of all intermediate encodings per layer
            Returned only if BERTEncoder.output_attention is True.
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
        super(BERTModel, self).__init__(prefix=prefix, params=params)
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

    def forward(self, inputs, token_types, valid_length=None, masked_positions=None):  # pylint: disable=arguments-differ
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
            decoder_out = self._decode(output, masked_positions)
            outputs.append(decoder_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def _encode_sequence(self, inputs, token_types, valid_length=None):
        """Generate the representation given the input sequences.

        This is used for pre-training or fine-tuning a BERT model.
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


model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('5656dac6965b5054147b0375337d5a6a7a2ff832', 'bert_12_768_12_book_corpus_wiki_en_cased'),
        ('75cc780f085e8007b3bf6769c6348bb1ff9a3074', 'bert_12_768_12_book_corpus_wiki_en_uncased'),
        ('237f39851b24f0b56d70aa20efd50095e3926e26', 'bert_12_768_12_wiki_multilingual'),
        ('237f39851b24f0b56d70aa20efd50095e3926e26', 'bert_12_768_12_wiki_multilingual_uncased'),
        ('b0f57a207f85a7d361bb79de80756a8c9a4276f7', 'bert_12_768_12_wiki_multilingual_cased'),
        ('885ebb9adc249a170c5576e90e88cfd1bbd98da6', 'bert_12_768_12_wiki_cn'),
        ('885ebb9adc249a170c5576e90e88cfd1bbd98da6', 'bert_12_768_12_wiki_cn_cased'),
        ('4e685a966f8bf07d533bd6b0e06c04136f23f620', 'bert_24_1024_16_book_corpus_wiki_en_cased'),
        ('24551e1446180e045019a87fc4ffbf714d99c0b5', 'bert_24_1024_16_book_corpus_wiki_en_uncased')
    ]})

bert_12_768_12_hparams = {
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
}

bert_24_1024_16_hparams = {
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
}

bert_hparams = {
    'bert_12_768_12': bert_12_768_12_hparams,
    'bert_24_1024_16': bert_24_1024_16_hparams,
}


def bert_12_768_12(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                   root=os.path.join(get_home_dir(), 'models'), use_pooler=True,
                   use_decoder=True, use_classifier=True, **kwargs):
    """Generic BERT BASE model.

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

    Returns
    -------
    BERTModel, gluonnlp.vocab.BERTVocab
    """
    return get_bert_model(model_name='bert_12_768_12', vocab=vocab,
                          dataset_name=dataset_name, pretrained=pretrained, ctx=ctx,
                          use_pooler=use_pooler, use_decoder=use_decoder,
                          use_classifier=use_classifier, root=root, **kwargs)


def bert_24_1024_16(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                    use_pooler=True, use_decoder=True, use_classifier=True,
                    root=os.path.join(get_home_dir(), 'models'), **kwargs):
    """Generic BERT LARGE model.

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

    Returns
    -------
    BERTModel, gluonnlp.vocab.BERTVocab
    """
    return get_bert_model(model_name='bert_24_1024_16', vocab=vocab,
                          dataset_name=dataset_name, pretrained=pretrained,
                          ctx=ctx, use_pooler=use_pooler,
                          use_decoder=use_decoder, use_classifier=use_classifier,
                          root=root, **kwargs)


def get_bert_model(model_name=None, dataset_name=None, vocab=None,
                   pretrained=True, ctx=mx.cpu(),
                   use_pooler=True, use_decoder=True, use_classifier=True,
                   output_attention=False, output_all_encodings=False,
                   root=os.path.join(get_home_dir(), 'models'), **kwargs):
    """Any BERT pretrained model.

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

    Returns
    -------
    BERTModel, gluonnlp.vocab.BERTVocab
    """
    predefined_args = bert_hparams[model_name]
    mutable_args = ['use_residual', 'dropout', 'embed_dropout', 'word_embed']
    mutable_args = frozenset(mutable_args)
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
        'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    # encoder
    encoder = BERTEncoder(attention_cell=predefined_args['attention_cell'],
                          num_layers=predefined_args['num_layers'],
                          units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'],
                          scaled=predefined_args['scaled'],
                          dropout=predefined_args['dropout'],
                          output_attention=output_attention,
                          output_all_encodings=output_all_encodings,
                          use_residual=predefined_args['use_residual'])
    # bert_vocab
    from ..vocab import BERTVocab
    if dataset_name in ['wiki_cn', 'wiki_multilingual']:
        warnings.warn('wiki_cn/wiki_multilingual will be deprecated.'
                      ' Please use wiki_cn_cased/wiki_multilingual_uncased instead.')
    bert_vocab = _load_vocab(dataset_name, vocab, root, cls=BERTVocab)
    # BERT
    net = BERTModel(encoder, len(bert_vocab),
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
