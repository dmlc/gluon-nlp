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
           'BERTLayerNorm', 'bert_12_768_12', 'bert_24_1024_16']

import os
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import model_store
import mxnet as mx
from .transformer import BasePositionwiseFFN, BaseTransformerEncoderCell, BaseTransformerEncoder
from .block import GELU
from .utils import _load_vocab, _load_pretrained_params

###############################################################################
#                              COMPONENTS                                     #
###############################################################################

class BERTLayerNorm(HybridBlock):
    """BERT style Layer Normalization.

    Epsilon is added inside the square root.

    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, epsilon=1e-12, in_channels=0, prefix=None, params=None):
        super(BERTLayerNorm, self).__init__(prefix=prefix, params=params)
        self.gamma = self.params.get('gamma', shape=(in_channels,),
                                     allow_deferred_init=True)
        self.beta = self.params.get('beta', shape=(in_channels,),
                                    allow_deferred_init=True)
        self._eps = epsilon

    def hybrid_forward(self, F, x, gamma, beta): # pylint: disable=arguments-differ
        u = F.mean(x, -1, keepdims=True)
        s = F.mean(F.broadcast_sub(x, u) ** 2, -1, keepdims=True) + self._eps
        x = F.broadcast_div(F.broadcast_sub(x, u), s.sqrt())
        return F.broadcast_add(F.broadcast_mul(gamma, x), beta)

    def __repr__(self):
        s = '{name}('
        in_channels = self.gamma.shape[0]
        s += 'in_channels={0}, epsilon={1})'.format(in_channels, self._eps)
        return s.format(name=self.__class__.__name__)

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
        super(BERTEncoder, self).__init__(attention_cell=attention_cell,
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
    """Model for BERT (Bidirectional Encoder Representations from Transformers).

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
        - **inputs**: input sequence tensor of shape (batch_size, seq_length)
        - **token_types**: input token type tensor of shape (batch_size, seq_length).
            If the inputs contain two sequences, then the token type of the first
            sequence differs from that of the second one.
        - **valid_length**: tensor for valid length of shape (batch_size)

    Outputs:
        - **sequence_outputs**: output tensor of sequence encodings.
            Shape (batch_size, seq_length, units).
        - **pooled_output**: output tensor of pooled representation of the first tokens.
            Returned only if use_pooler is True. Shape (batch_size, units)
        - **classifier_output**: output tensor of next sentence classification prediction.
            Returned only if use_classifier is True. Shape (batch_size, 2)
        - **decode_output**: output tensor of sequence decoding for masked language model
            prediction. Returned only if use_decoder True.
            Shape (batch_size, vocab_size)
    """
    def __init__(self, encoder, vocab_size=None, token_type_vocab_size=None, units=None,
                 embed_size=None, embed_dropout=0.0, embed_initializer=None,
                 word_embed=None, token_type_embed=None, use_pooler=True, use_decoder=True,
                 use_classifier=True, prefix=None, params=None):
        super(BERTModel, self).__init__(prefix=prefix, params=params)
        self._use_decoder = use_decoder
        self._use_classifier = use_classifier
        self._use_pooler = use_pooler
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
            self.decoder = self._get_decoder(units, vocab_size, self.word_embed, 'decoder_')

    def _get_classifier(self, prefix):
        """ Construct a decoder for the masked language model task """
        with self.name_scope():
            classifier = nn.Dense(2, prefix=prefix)
        return classifier

    def _get_decoder(self, units, vocab_size, embed, prefix):
        """ Construct a decoder for the masked language model task """
        with self.name_scope():
            decoder = nn.HybridSequential(prefix=prefix)
            decoder.add(nn.Dense(units))
            decoder.add(GELU())
            decoder.add(BERTLayerNorm(in_channels=units))
            decoder.add(nn.Dense(vocab_size, params=embed.params))
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

    def forward(self, inputs, token_types, valid_length=None): #pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        outputs = []
        seq_out, _ = self._encode_sequence(inputs, token_types, valid_length)
        outputs.append(seq_out)
        if self._use_pooler:
            pooled_out = self._apply_pooling(seq_out)
            outputs.append(pooled_out)
            if self._use_classifier:
                classifier_out = self.classifier(pooled_out)
                outputs.append(classifier_out)
        if self._use_decoder:
            decoder_out = self._decode(seq_out, valid_length)
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

    def _decode(self, sequence, valid_length=None):
        """Generate unormalized prediction for the masked language model task.

        This is only used for pre-training the BERT model.
        """
        if valid_length is None:
            last_step = sequence[:, -1, :]
        else:
            batch_size = sequence.shape[0]
            ctx = valid_length.context
            dtype = valid_length.dtype
            batch_idx = mx.nd.arange(0, batch_size, dtype=dtype, ctx=ctx).reshape((1, -1))
            valid_length = (valid_length - 1).reshape((1, -1))
            last_step_idx = mx.nd.Concat(batch_idx, valid_length, dim=0)
            last_step = mx.nd.gather_nd(sequence, last_step_idx)
        return self.decoder(last_step)

###############################################################################
#                               GET MODEL                                     #
###############################################################################

model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('5656dac6965b5054147b0375337d5a6a7a2ff832', 'bert_12_768_12_book_corpus_wiki_en_cased'),
        ('75cc780f085e8007b3bf6769c6348bb1ff9a3074', 'bert_12_768_12_book_corpus_wiki_en_uncased'),
        ('237f39851b24f0b56d70aa20efd50095e3926e26', 'bert_12_768_12_wiki_multilingual'),
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
                   root=os.path.join('~', '.mxnet', 'models'), use_pooler=True,
                   use_decoder=True, use_classifier=True, **kwargs):
    """BERT BASE pretrained model.

    The number of layers (L) is 12, number of units (H) is 768, and the
    number of self-attention heads (A) is 12.

    Parameters
    ----------
    dataset_name : str or None, default None
        Options include 'book_corpus_wiki_en_cased', 'book_corpus_wiki_en_uncased',
        and 'wiki_multilingual'.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
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
    BERTModel, gluonnlp.Vocab
    """
    return _bert_model(model_name='bert_12_768_12', vocab=vocab, dataset_name=dataset_name,
                       pretrained=pretrained, ctx=ctx, use_pooler=use_pooler,
                       use_decoder=use_decoder, use_classifier=use_classifier, root=root,
                       **kwargs)

def bert_24_1024_16(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                    use_pooler=True, use_decoder=True, use_classifier=True,
                    root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """BERT LARGE pretrained model.

    The number of layers (L) is 24, number of units (H) is 1024, and the
    number of self-attention heads (A) is 16.

    Parameters
    ----------
    dataset_name : str or None, default None
        Options include 'book_corpus_wiki_en_uncased'.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset is not specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
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
    BERTModel, gluonnlp.Vocab
    """
    return _bert_model(model_name='bert_24_1024_16', vocab=vocab, dataset_name=dataset_name,
                       pretrained=pretrained, ctx=ctx, use_pooler=use_pooler,
                       use_decoder=use_decoder, use_classifier=use_classifier, root=root,
                       **kwargs)

def _bert_model(model_name=None, dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                use_pooler=True, use_decoder=True, use_classifier=True,
                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """BERT pretrained model.

    Returns
    -------
    BERTModel, gluonnlp.Vocab
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
                          use_residual=predefined_args['use_residual'])
    # vocab
    vocab = _load_vocab(dataset_name, vocab, root)
    # BERT
    net = BERTModel(encoder, len(vocab),
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
    return net, vocab
