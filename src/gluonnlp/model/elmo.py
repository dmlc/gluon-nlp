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

"""ELMo."""
__all__ = ['ELMoBiLM', 'ELMoCharacterEncoder', 'ELMoCharacterVocab',
           'elmo_2x1024_128_2048cnn_1xhighway', 'elmo_2x2048_256_2048cnn_1xhighway',
           'elmo_2x4096_512_2048cnn_2xhighway']

import os
import numpy
import mxnet as mx

from mxnet import gluon, nd, cpu
from mxnet.gluon.model_zoo import model_store
from mxnet.gluon.model_zoo.model_store import get_model_file
try:
    from convolutional_encoder import ConvolutionalEncoder
    from bilm_encoder import BiLMEncoder
    from initializer.initializer import HighwayBias
except ImportError:
    from .convolutional_encoder import ConvolutionalEncoder
    from .bilm_encoder import BiLMEncoder
    from ..initializer.initializer import HighwayBias

def _make_bos_eos(
        character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


def _add_sentence_boundary_token_ids(inputs, mask, sentence_begin_token, sentence_end_token):
    sequence_lengths = mask.sum(axis=1).asnumpy()
    inputs_shape = list(inputs.shape)
    new_shape = list(inputs_shape)
    new_shape[1] = inputs_shape[1] + 2
    inputs_with_boundary_tokens = nd.zeros(new_shape)
    if len(inputs_shape) == 2:
        inputs_with_boundary_tokens[:, 1:-1] = inputs
        inputs_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            inputs_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = inputs_with_boundary_tokens != 0
    elif len(inputs_shape) == 3:
        inputs_with_boundary_tokens[:, 1:-1, :] = inputs
        for i, j in enumerate(sequence_lengths):
            inputs_with_boundary_tokens[i, 0, :] = sentence_begin_token
            inputs_with_boundary_tokens[i, int(j + 1), :] = sentence_end_token
        new_mask = (inputs_with_boundary_tokens > 0).sum(axis=-1) > 0
    else:
        raise NotImplementedError
    return inputs_with_boundary_tokens, new_mask


class ELMoCharacterVocab:
    r"""ELMo special character vocabulary

    The vocab aims to map individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here.

    Specifically, char ids 0-255 come from utf-8 encoding bytes.
    Above 256 are reserved for special tokens.

    Parameters
    ----------
    max_word_length: 50
        The maximum number of character a word contains is 50 in ELMo.
    beginning_of_sentence_character: 256
        The index of beginning of the sentence character is 256 in ELMo.
    end_of_sentence_character: 257
        The index of end of the sentence character is 257 in ELMo.
    beginning_of_word_character : 258
        The index of beginning of the sentence character is 258 in ELMo.
    end_of_word_character: 259
        The index of end of the sentence character is 259 in ELMo.
    padding_character: 260
        The index of padding character is 260 in ELMo.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # bos
    beginning_of_sentence_character = 256
    # eos
    end_of_sentence_character = 257
    # begin of word
    beginning_of_word_character = 258
    # <end of word>
    end_of_word_character = 259
    # <padding>
    padding_character = 260

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length
    )

    bos_token = '<bos>'
    eos_token = '<eos>'


class ELMoCharacterEncoder(gluon.Block):
    r"""ELMo character encoder

    Compute context insensitive character based token representation using pretrained biLM.

    This encoder has input character ids of shape
    (batch_size, sequence_length, max_character_per_word)
    and returns (batch_size, sequence_length + 2, embedding_size).

    We add special entries at the beginning and end of each sequence corresponding
    to <bos> and <eos>, the beginning and end of sentence tokens.

    Parameters
    ----------
    output_size: int
        The output dimension after conducting the convolutions and max pooling,
        and applying highways, as well as linear projection.
    filters: List[List]
        The first elements in the list are:
        ngram_filter_sizes:
        The size of each convolutional layer,
        and len(ngram_filter_sizes) equals to the number of convolutional layers.
        The second elements in the list are:
        num_filters:
        The output dimension for each convolutional layer according to the filter sizes,
        which are the number of the filters learned by the layers.
    char_embed_size : int
        The input dimension to the encoder.
    num_highway: int
        The number of layers of the Highway layer.
    conv_layer_activation: str
        Activation function to be used after convolutional layer.
    max_chars_per_token: int
        The maximum number of characters of a token.
    """
    def __init__(self,
                 output_size,
                 filters,
                 char_embed_size,
                 num_highway,
                 conv_layer_activation,
                 max_chars_per_token):
        super(ELMoCharacterEncoder, self).__init__()

        self._output_size = output_size
        # Cache the arrays when using mask.
        self._beginning_of_sentence_characters = mx.nd.array(
            numpy.array(ELMoCharacterVocab.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = mx.nd.array(
            numpy.array(ELMoCharacterVocab.end_of_sentence_characters) + 1
        )
        self._char_embed_size = char_embed_size
        self._filters = filters
        ngram_filter_sizes = []
        num_filters = []
        for _, (width, num) in enumerate(self._filters):
            ngram_filter_sizes.append(width)
            num_filters.append(num)
        self._num_highway = num_highway
        self._conv_layer_activation = conv_layer_activation
        self._max_chars_per_token = max_chars_per_token

        with self.name_scope():
            self._char_embedding = gluon.nn.Embedding(ELMoCharacterVocab.padding_character+2,
                                                      self._char_embed_size)
            self._convolutions = ConvolutionalEncoder(embed_size=self._char_embed_size,
                                                      num_filters=tuple(num_filters),
                                                      ngram_filter_sizes=tuple(ngram_filter_sizes),
                                                      conv_layer_activation=conv_layer_activation,
                                                      num_highway=self._num_highway,
                                                      highway_bias=HighwayBias(
                                                          nonlinear_transform_bias=0.0,
                                                          transform_gate_bias=1.0),
                                                      output_size=self._output_size)

    def get_output_size(self):
        return self._output_size

    def forward(self, inputs):
        # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs: ``NDArray``
            Shape ``(batch_size, sequence_length, max_character_per_token)``
            of character ids representing the current batch.

        Returns
        -------
        mask:  ``NDArray``
            Shape ``(batch_size, sequence_length + 2)`` with sequence mask.
        token_embedding: ``NDArray``
            Shape ``(batch_size, sequence_length + 2, embedding_size)`` with context
            insensitive token representations.
        """
        mask = (inputs > 0).sum(axis=-1) > 0
        character_ids_with_bos_eos, mask_with_bos_eos = _add_sentence_boundary_token_ids(
            inputs,
            mask,
            self._beginning_of_sentence_characters,
            self._end_of_sentence_characters
        )

        # the character id embedding
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = self._char_embedding(character_ids_with_bos_eos.
                                                   reshape(-1, self._max_chars_per_token))

        character_embedding = nd.transpose(character_embedding, axes=(1, 0, 2))
        token_embedding = self._convolutions(character_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.shape

        return mask_with_bos_eos, token_embedding.reshape(batch_size, sequence_length, -1)


class ELMoBiLM(gluon.Block):
    r"""ELMo Bidirectional language model

    Run a pre-trained bidirectional language model, outputing the weighted
    ELMo representation.

    We implement the ELMo Bidirectional language model (BiLm) proposed in the following work::

        @inproceedings{Peters:2018,
        author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner,
        Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
        title={Deep contextualized word representations},
        booktitle={Proc. of NAACL},
        year={2018}
        }

    Parameters
    ----------
    model : str
        The type of RNN cell to use. Option is 'lstmpc' due to the pretrained setting.
    output_size: int
        The output dimension after conducting the convolutions and max pooling,
        and applying highways, as well as linear projection.
    filters: List[List[int,int]]
        The first elements in the list are:
        ngram_filter_sizes:
        The size of each convolutional layer,
        and len(ngram_filter_sizes) equals to the number of convolutional layers.
        The second elements in the list are:
        num_filters:
        The output dimension for each convolutional layer according to the filter sizes,
        which are the number of the filters learned by the layers.
    char_embed_size : int
        The input dimension to the encoder.
    num_highway: int
        The number of layers of the Highway layer.
    conv_layer_activation: str
        Activation function to be used after convolutional layer.
    max_chars_per_token: int
        The maximum number of characters of a token.
    input_size : int
        The initial input size of in the RNN cell.
    hidden_size : int
        The hidden size of the RNN cell.
    proj_size : int
        The projection size of each LSTMPCellWithClip cell
    num_layers : int
        The number of RNN cells.
    cell_clip : float
        Clip cell state between [-cellclip, cell_clip] in LSTMPCellWithClip cell
    proj_clip : float
        Clip projection between [-projclip, projclip] in LSTMPCellWithClip cell
    skip_connection : bool
        Whether to add skip connections (add RNN cell input to output)
    """
    def __init__(self,
                 model,
                 output_size,
                 filters,
                 char_embed_size,
                 num_highway,
                 conv_layer_activation,
                 max_chars_per_token,
                 input_size,
                 hidden_size,
                 proj_size,
                 num_layers,
                 cell_clip,
                 proj_clip,
                 skip_connection=True):
        super(ELMoBiLM, self).__init__()

        self._model = model
        self._output_size = output_size
        self._filters = filters
        self._char_embed_size = char_embed_size
        self._num_highway = num_highway
        self._conv_layer_activation = conv_layer_activation
        self._max_chars_per_token = max_chars_per_token
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._proj_size = proj_size
        self._num_layers = num_layers
        self._cell_clip = cell_clip
        self._proj_clip = proj_clip
        self._skip_connection = skip_connection

        if not self._skip_connection:
            raise NotImplementedError

        with self.name_scope():
            self._elmo_char_encoder = ELMoCharacterEncoder(self._output_size,
                                                           self._filters,
                                                           self._char_embed_size,
                                                           self._num_highway,
                                                           self._conv_layer_activation,
                                                           self._max_chars_per_token)
            self._elmo_lstm = BiLMEncoder(mode=self._model,
                                          input_size=self._input_size,
                                          hidden_size=self._hidden_size,
                                          proj_size=self._proj_size,
                                          num_layers=self._num_layers,
                                          cell_clip=self._cell_clip,
                                          proj_clip=self._proj_clip)

    def get_output_size(self):
        return 2 * self._elmo_char_encoder.get_output_size()

    def begin_state(self, func, **kwargs):
        return self._elmo_lstm.begin_state(func, **kwargs)

    def forward(self, inputs, states=None):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        inputs: ``NDArray``
            Shape ``(batch_size, sequence_length, max_character_per_token)``
            of character ids representing the current batch.
        states : List[List[List[NDArray]]]
            The states. including:
            states[0] indicates the states used in forward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
            states[1] indicates the states used in backward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).

        Returns
        -------
        output: List[NDArray]
            A list of activations at each layer of the network, each of shape
            ``(batch_size, sequence_length + 2, embedding_size)``
        states : List[List[List[NDArray]]]
            The states. including:
            states[0] indicates the output states from forward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
            states[1] indicates the output states from backward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
        mask:  ``NDArray``
            Shape ``(batch_size, sequence_length + 2)`` with sequence mask.
        """

        mask, type_representation = self._elmo_char_encoder(inputs)
        type_representation = type_representation.transpose(axes=(1, 0, 2))
        lstm_outputs, states = self._elmo_lstm(type_representation, states, mask)
        lstm_outputs = lstm_outputs.transpose(axes=(0, 2, 1, 3))
        type_representation = type_representation.transpose(axes=(1, 0, 2))

        # Prepare the output. The first layer is duplicated.
        output = [
            mx.nd.concat(*[type_representation, type_representation], dim=-1)
            * mask.expand_dims(axis=-1)
        ]
        for layer_activations in mx.nd.split(lstm_outputs, lstm_outputs.shape[0], axis=0):
            output.append(layer_activations.squeeze(axis=0))
        return output, states, mask


model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('8c9257d9153436e9eb692f9ec48d8ee07e2120f8', 'elmo_2x1024_128_2048cnn_1xhighway_gbw'),
        ('85eab56a3c90c6866dd8d13b50449934be58a2e6', 'elmo_2x2048_256_2048cnn_1xhighway_gbw'),
        ('79af623840c13b10cb891d20c207afc483ab27b9', 'elmo_2x4096_512_2048cnn_2xhighway_5bw'),
        ('5608a09f33c52e5ab3f043b1793481ab448a0347', 'elmo_2x4096_512_2048cnn_2xhighway_gbw')
    ]})


def _load_pretrained_params(net, model_name, dataset_name, root, ctx):
    model_file = get_model_file('_'.join([model_name, dataset_name]), root=root)
    net.load_parameters(model_file, ctx=ctx)


def _get_elmo_model(model_cls, model_name, dataset_name, pretrained, ctx, root, **kwargs):
    net = model_cls(**kwargs)
    if pretrained:
        _load_pretrained_params(net, model_name, dataset_name, root, ctx)
    return net


def elmo_2x1024_128_2048cnn_1xhighway(dataset_name=None, pretrained=False, ctx=cpu(),
                                      root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""ELMo 2-layer BiLSTM with 1024 hidden units, 128 projection size, 1 highway layer.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pre-trained model is trained.
        Options are 'gbw'.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block
    """

    predefined_args = {'model': 'lstmpc',
                       'output_size': 128,
                       'filters': [[1, 32], [2, 32], [3, 64], [4, 128],
                                   [5, 256], [6, 512], [7, 1024]],
                       'char_embed_size': 16,
                       'num_highway': 1,
                       'conv_layer_activation': 'relu',
                       'max_chars_per_token': 50,
                       'input_size': 128,
                       'hidden_size': 1024,
                       'proj_size': 128,
                       'num_layers': 2,
                       'cell_clip': 3,
                       'proj_clip': 3,
                       'skip_connection': True}
    assert all((k not in kwargs) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    return _get_elmo_model(ELMoBiLM, 'elmo_2x1024_128_2048cnn_1xhighway', dataset_name, pretrained,
                           ctx, root, **predefined_args)


def elmo_2x2048_256_2048cnn_1xhighway(dataset_name=None, pretrained=False, ctx=cpu(),
                                      root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""ELMo 2-layer BiLSTM with 2048 hidden units, 256 projection size, 1 highway layer.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pre-trained model is trained.
        Options are 'gbw'.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block
    """

    predefined_args = {'model': 'lstmpc',
                       'output_size': 256,
                       'filters': [[1, 32], [2, 32], [3, 64], [4, 128],
                                   [5, 256], [6, 512], [7, 1024]],
                       'char_embed_size': 16,
                       'num_highway': 1,
                       'conv_layer_activation': 'relu',
                       'max_chars_per_token': 50,
                       'input_size': 256,
                       'hidden_size': 2048,
                       'proj_size': 256,
                       'num_layers': 2,
                       'cell_clip': 3,
                       'proj_clip': 3,
                       'skip_connection': True}
    assert all((k not in kwargs) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    return _get_elmo_model(ELMoBiLM, 'elmo_2x2048_256_2048cnn_1xhighway', dataset_name, pretrained,
                           ctx, root, **predefined_args)


def elmo_2x4096_512_2048cnn_2xhighway(dataset_name=None, pretrained=False, ctx=cpu(),
                                      root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""ELMo 2-layer BiLSTM with 4096 hidden units, 512 projection size, 2 highway layer.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pre-trained model is trained.
        Options are 'gbw' and '5bw'.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block
    """

    predefined_args = {'model': 'lstmpc',
                       'output_size': 512,
                       'filters': [[1, 32], [2, 32], [3, 64], [4, 128],
                                   [5, 256], [6, 512], [7, 1024]],
                       'char_embed_size': 16,
                       'num_highway': 2,
                       'conv_layer_activation': 'relu',
                       'max_chars_per_token': 50,
                       'input_size': 512,
                       'hidden_size': 4096,
                       'proj_size': 512,
                       'num_layers': 2,
                       'cell_clip': 3,
                       'proj_clip': 3,
                       'skip_connection': True}
    assert all((k not in kwargs) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    return _get_elmo_model(ELMoBiLM, 'elmo_2x4096_512_2048cnn_2xhighway', dataset_name, pretrained,
                           ctx, root, **predefined_args)
