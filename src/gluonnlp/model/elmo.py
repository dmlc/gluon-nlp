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
__all__ = ['ELMoBiLM', 'ELMoCharacterEncoder',
           'elmo_2x1024_128_2048cnn_1xhighway', 'elmo_2x2048_256_2048cnn_1xhighway',
           'elmo_2x4096_512_2048cnn_2xhighway']

import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import model_store
from mxnet.gluon.model_zoo.model_store import get_model_file

from .convolutional_encoder import ConvolutionalEncoder
from .bilm_encoder import BiLMEncoder
from ..initializer.initializer import HighwayBias
from ..vocab.elmo import ELMoCharVocab


class ELMoCharacterEncoder(gluon.HybridBlock):
    r"""ELMo character encoder

    Compute context-free character-based token representation with character-level convolution.

    This encoder has input character ids of shape
    (batch_size, sequence_length, max_character_per_word)
    and returns (batch_size, sequence_length, embedding_size).

    Parameters
    ----------
    output_size : int
        The output dimension after conducting the convolutions and max pooling,
        and applying highways, as well as linear projection.
    filters : list of tuple
        List of tuples representing the settings for convolution layers.
        Each element is (ngram_filter_size, num_filters).
    char_embed_size : int
        The input dimension to the encoder.
    num_highway : int
        The number of layers of the Highway layer.
    conv_layer_activation : str
        Activation function to be used after convolutional layer.
    max_chars_per_token : int
        The maximum number of characters of a token.
    char_vocab_size : int
        Size of character-level vocabulary.
    """
    def __init__(self,
                 output_size,
                 filters,
                 char_embed_size,
                 num_highway,
                 conv_layer_activation,
                 max_chars_per_token,
                 char_vocab_size,
                 **kwargs):
        super(ELMoCharacterEncoder, self).__init__(**kwargs)

        self._output_size = output_size
        self._char_embed_size = char_embed_size
        self._filters = filters
        ngram_filter_sizes = []
        num_filters = []
        for width, num in filters:
            ngram_filter_sizes.append(width)
            num_filters.append(num)
        self._num_highway = num_highway
        self._conv_layer_activation = conv_layer_activation
        self._max_chars_per_token = max_chars_per_token

        with self.name_scope():
            self._char_embedding = gluon.nn.Embedding(char_vocab_size,
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


    def hybrid_forward(self, F, inputs):
        # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs : NDArray
            Shape (batch_size, sequence_length, max_character_per_token)
            of character ids representing the current batch.

        Returns
        -------
        token_embedding : NDArray
            Shape (batch_size, sequence_length, embedding_size) with context
            insensitive token representations.
        """
        # the character id embedding
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = self._char_embedding(inputs.reshape((-1, self._max_chars_per_token)))

        character_embedding = F.transpose(character_embedding, axes=(1, 0, 2))
        token_embedding = self._convolutions(character_embedding)

        out_shape_ref = inputs.slice_axis(axis=-1, begin=0, end=1)
        out_shape_ref = out_shape_ref.broadcast_axes(axis=(2,),
                                                     size=(self._output_size))

        return token_embedding.reshape_like(out_shape_ref)


class ELMoBiLM(gluon.HybridBlock):
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
    rnn_type : str
        The type of RNN cell to use.
        The option for pre-trained models is 'lstmpc'.
    output_size : int
        The output dimension after conducting the convolutions and max pooling,
        and applying highways, as well as linear projection.
    filters : list of tuple
        List of tuples representing the settings for convolution layers.
        Each element is (ngram_filter_size, num_filters).
    char_embed_size : int
        The input dimension to the encoder.
    char_vocab_size : int
        Size of character-level vocabulary.
    num_highway : int
        The number of layers of the Highway layer.
    conv_layer_activation : str
        Activation function to be used after convolutional layer.
    max_chars_per_token : int
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
                 rnn_type,
                 output_size,
                 filters,
                 char_embed_size,
                 char_vocab_size,
                 num_highway,
                 conv_layer_activation,
                 max_chars_per_token,
                 input_size,
                 hidden_size,
                 proj_size,
                 num_layers,
                 cell_clip,
                 proj_clip,
                 skip_connection=True,
                 **kwargs):
        super(ELMoBiLM, self).__init__(**kwargs)

        self._rnn_type = rnn_type
        self._output_size = output_size
        self._filters = filters
        self._char_embed_size = char_embed_size
        self._char_vocab_size = char_vocab_size
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
                                                           self._max_chars_per_token,
                                                           self._char_vocab_size)
            self._elmo_lstm = BiLMEncoder(mode=self._rnn_type,
                                          input_size=self._input_size,
                                          hidden_size=self._hidden_size,
                                          proj_size=self._proj_size,
                                          num_layers=self._num_layers,
                                          cell_clip=self._cell_clip,
                                          proj_clip=self._proj_clip)

    def begin_state(self, func, **kwargs):
        return self._elmo_lstm.begin_state(func, **kwargs)

    def hybrid_forward(self, F, inputs, states=None, mask=None):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        inputs : NDArray
            Shape (batch_size, sequence_length, max_character_per_token)
            of character ids representing the current batch.
        states : (list of list of NDArray, list of list of NDArray)
            The states. First tuple element is the forward layer states, while the second is
            the states from backward layer. Each is a list of states for each layer.
            The state of each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
        mask :  NDArray
            Shape (batch_size, sequence_length) with sequence mask.

        Returns
        -------
        output : list of NDArray
            A list of activations at each layer of the network, each of shape
            (batch_size, sequence_length, embedding_size)
        states : (list of list of NDArray, list of list of NDArray)
            The states. First tuple element is the forward layer states, while the second is
            the states from backward layer. Each is a list of states for each layer.
            The state of each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
        """

        type_representation = self._elmo_char_encoder(inputs)
        type_representation = type_representation.transpose(axes=(1, 0, 2))
        lstm_outputs, states = self._elmo_lstm(type_representation, states, mask)
        lstm_outputs = lstm_outputs.transpose(axes=(0, 2, 1, 3))
        type_representation = type_representation.transpose(axes=(1, 0, 2))

        # Prepare the output. The first layer is duplicated.
        output = F.concat(*[type_representation, type_representation], dim=-1)
        if mask is not None:
            output = output * mask.expand_dims(axis=-1)
        output = [output]
        output.extend([layer_activations.squeeze(axis=0) for layer_activations
                       in F.split(lstm_outputs, self._num_layers, axis=0)])
        return output, states


model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('8c9257d9153436e9eb692f9ec48d8ee07e2120f8', 'elmo_2x1024_128_2048cnn_1xhighway_gbw'),
        ('85eab56a3c90c6866dd8d13b50449934be58a2e6', 'elmo_2x2048_256_2048cnn_1xhighway_gbw'),
        ('79af623840c13b10cb891d20c207afc483ab27b9', 'elmo_2x4096_512_2048cnn_2xhighway_5bw'),
        ('5608a09f33c52e5ab3f043b1793481ab448a0347', 'elmo_2x4096_512_2048cnn_2xhighway_gbw')
    ]})



def _get_elmo_model(model_cls, model_name, dataset_name, pretrained, ctx, root, **kwargs):
    vocab = ELMoCharVocab()
    if 'char_vocab_size' not in kwargs:
        kwargs['char_vocab_size'] = len(vocab)
    net = model_cls(**kwargs)
    if pretrained:
        model_file = get_model_file('_'.join([model_name, dataset_name]), root=root)
        net.load_parameters(model_file, ctx=ctx)
    return net, vocab


def elmo_2x1024_128_2048cnn_1xhighway(dataset_name=None, pretrained=False, ctx=mx.cpu(),
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

    predefined_args = {'rnn_type': 'lstmpc',
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


def elmo_2x2048_256_2048cnn_1xhighway(dataset_name=None, pretrained=False, ctx=mx.cpu(),
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

    predefined_args = {'rnn_type': 'lstmpc',
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


def elmo_2x4096_512_2048cnn_2xhighway(dataset_name=None, pretrained=False, ctx=mx.cpu(),
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

    predefined_args = {'rnn_type': 'lstmpc',
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
