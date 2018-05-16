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
__all__ = ['AWDRNN', 'StandardRNN', 'forward', 'detach', 'user_pretrained_lm', 'awd_lstm_lm_1150', 'awd_lstm_lm_600',
           'standard_lstm_lm_200', 'standard_lstm_lm_650', 'standard_lstm_lm_1500']

import os
import warnings

from mxnet.gluon.model_zoo.model_store import get_model_file
from mxnet import init, nd, cpu, autograd
from mxnet.gluon import nn, Block
from mxnet.gluon.model_zoo import model_store

from .utils import _get_rnn_layer
from .utils import apply_weight_drop
from ..data.utils import _load_pretrained_vocab


class AWDRNN(Block):
    """AWD language model by salesforce.

    Reference: https://github.com/salesforce/awd-lstm-lm

    License: BSD 3-Clause

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
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
    """
    def __init__(self, mode, vocab_size, embed_size=400, hidden_size=1150, num_layers=3,
                 tie_weights=True, dropout=0.4, weight_drop=0.5, drop_h=0.2, drop_i=0.65, drop_e=0.1,
                 **kwargs):
        super(AWDRNN, self).__init__(**kwargs)
        self._mode = mode
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding_block = nn.Embedding(self._vocab_size, self._embed_size,
                                           weight_initializer=init.Uniform(0.1))
            if self._drop_e:
                apply_weight_drop(embedding_block, 'weight', self._drop_e, axes=(1,))
            embedding.add(embedding_block)
            if self._drop_i:
                embedding.add(nn.Dropout(self._drop_i, axes=(0,)))
        return embedding

    def _get_encoder(self):
        encoder = nn.Sequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(_get_rnn_layer(self._mode, 1, self._embed_size if l == 0 else
                                           self._hidden_size, self._hidden_size if
                                           l != self._num_layers - 1 or not self._tie_weights
                                           else self._embed_size, 0, self._weight_drop))
        return encoder

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return [c.begin_state(*args, **kwargs) for c in self.encoder]

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Implement forward computation.

        Parameters
        ----------
        inputs : NDArray
            The training dataset.
        begin_state : list
            The initial hidden states.

        Returns
        -------
        out: NDArray
            The output of the model.
        out_states: list
            The list of output states of the model's encoder.
        """
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        out_states = []
        for i, (e, s) in enumerate(zip(self.encoder, begin_state)):
            encoded, state = e(encoded, s)
            out_states.append(state)
            if self._drop_h and i != len(self.encoder)-1:
                encoded = nd.Dropout(encoded, p=self._drop_h, axes=(0,))
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        with autograd.predict_mode():
            out = self.decoder(encoded)
        return out, out_states


class StandardRNN(Block):
    """Standard RNN language model.

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    dropout : float
        Dropout rate to use for encoder output.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    """
    def __init__(self, mode, vocab_size, embed_size, hidden_size,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        if tie_weights:
            assert embed_size == hidden_size, 'Embedding dimension must be equal to ' \
                                              'hidden dimension in order to tie weights. ' \
                                              'Got: emb: {}, hid: {}.'.format(embed_size,
                                                                              hidden_size)
        super(StandardRNN, self).__init__(**kwargs)
        self._mode = mode
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._tie_weights = tie_weights
        self._vocab_size = vocab_size

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(nn.Embedding(self._vocab_size, self._embed_size,
                                       weight_initializer=init.Uniform(0.1)))
            if self._dropout:
                embedding.add(nn.Dropout(self._dropout))
        return embedding

    def _get_encoder(self):
        return _get_rnn_layer(self._mode, self._num_layers, self._embed_size,
                              self._hidden_size, self._dropout, 0)

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        encoded = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        encoded, state = self.encoder(encoded, begin_state)
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        out = self.decoder(encoded)
        return out, state

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def forward(model, inputs, begin_state=None):
    """Implement the forward computation that the awd language model and cache model use.

    Parameters
    ----------
    inputs : NDArray
        The training dataset.
    begin_state : list
        The initial hidden states.

    Returns
    -------
    out: NDArray
        The output of the model.
    out_states: list
        The list of output states of the model's encoder.
    encoded_raw: list
        The list of outputs of the model's encoder.
    encoded_dropped: list
        The list of outputs with dropout of the model's encoder.
    """
    encoded = model.embedding(inputs)
    if not begin_state:
        begin_state = model.begin_state(batch_size=inputs.shape[1])
    out_states = []
    encoded_raw = []
    encoded_dropped = []
    if 'awd' in model.prefix:
        for i, (e, s) in enumerate(zip(model.encoder, begin_state)):
            encoded, state = e(encoded, s)
            encoded_raw.append(encoded)
            out_states.append(state)
            if model._drop_h and i != len(model.encoder) - 1:
                encoded = nd.Dropout(encoded, p=model._drop_h, axes=(0,))
                encoded_dropped.append(encoded)
    else:
        encoded, state = model.encoder(encoded, begin_state)
        encoded_raw.append(encoded)
    if model._dropout:
        encoded = nd.Dropout(encoded, p=model._dropout, axes=(0,))
    if 'awd' in model.prefix:
        encoded_dropped.append(encoded)
        with autograd.predict_mode():
            out = model.decoder(encoded)
    else:
        out = model.decoder(encoded)
    if 'awd' in model.prefix:
        return out, out_states, encoded_raw, encoded_dropped
    else:
        return out, state, encoded_raw, encoded_dropped

def _load_vocab(dataset_name, vocab, root):
    if dataset_name:
        if vocab is not None:
            warnings.warn('Both dataset_name and vocab are specified. Loading vocab for dataset. '
                          'Input "vocab" argument will be ignored.')
        vocab = _load_pretrained_vocab(dataset_name, root)
    else:
        assert vocab is not None, 'Must specify vocab if not loading from predefined datasets.'
    return vocab


def _load_pretrained_params(net, model_name, dataset_name, root, ctx):
    model_file = get_model_file('_'.join([model_name, dataset_name]), root=root)
    net.load_params(model_file, ctx=ctx)


def _get_rnn_model(model_cls, model_name, dataset_name, vocab, pretrained, ctx, root, **kwargs):
    vocab = _load_vocab(dataset_name, vocab, root)
    kwargs['vocab_size'] = len(vocab)
    net = model_cls(**kwargs)
    if pretrained:
        _load_pretrained_params(net, model_name, dataset_name, root, ctx)
    return net, vocab


def awd_lstm_lm_1150(dataset_name=None, vocab=None, pretrained=False, ctx=cpu(),
                     root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""3-layer LSTM language model with weight-drop, variational dropout, and tied weights.

    Embedding size is 400, and hidden layer size is 1150.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pretrained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
        The pre-trained model is not provided yet.
    vocab : gluonnlp.Vocab or None, default None
        Vocab object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab
    """
    predefined_args = {'embed_size': 400,
                       'hidden_size': 1150,
                       'mode': 'lstm',
                       'num_layers': 3,
                       'tie_weights': True,
                       'dropout': 0.4,
                       'weight_drop': 0.5,
                       'drop_h': 0.2,
                       'drop_i': 0.65,
                       'drop_e': 0.1}
    mutable_args = frozenset(['dropout', 'weight_drop', 'drop_h', 'drop_i', 'drop_e'])
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    return _get_rnn_model(AWDRNN, 'awd_lstm_lm_1150', dataset_name, vocab, pretrained,
                          ctx, root, **predefined_args)


def awd_lstm_lm_600(dataset_name=None, vocab=None, pretrained=False, ctx=cpu(),
                    root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""3-layer LSTM language model with weight-drop, variational dropout, and tied weights.

    Embedding size is 200, and hidden layer size is 600.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pretrained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
        The pre-trained model is not provided yet.
    vocab : gluonnlp.Vocab or None, default None
        Vocab object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab
    """
    predefined_args = {'embed_size': 200,
                       'hidden_size': 600,
                       'mode': 'lstm',
                       'num_layers': 3,
                       'tie_weights': True,
                       'dropout': 0.2,
                       'weight_drop': 0.2,
                       'drop_h': 0.1,
                       'drop_i': 0.3,
                       'drop_e': 0.05}
    mutable_args = frozenset(['dropout', 'weight_drop', 'drop_h', 'drop_i', 'drop_e'])
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    return _get_rnn_model(AWDRNN, 'awd_lstm_lm_600', dataset_name, vocab, pretrained,
                          ctx, root, **predefined_args)

def standard_lstm_lm_200(dataset_name=None, vocab=None, pretrained=False, ctx=cpu(),
                         root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Standard 2-layer LSTM language model with tied embedding and output weights.

    Both embedding and hidden dimensions are 200.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pretrained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
        The pre-trained model achieves 102.91 ppl on wikitext-2.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab
    """
    predefined_args = {'embed_size': 200,
                       'hidden_size': 200,
                       'mode': 'lstm',
                       'num_layers': 2,
                       'tie_weights': True,
                       'dropout': 0.2}
    mutable_args = ['dropout']
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    return _get_rnn_model(StandardRNN, 'standard_lstm_lm_200', dataset_name, vocab, pretrained,
                          ctx, root, **predefined_args)


def standard_lstm_lm_650(dataset_name=None, vocab=None, pretrained=False, ctx=cpu(),
                         root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Standard 2-layer LSTM language model with tied embedding and output weights.

    Both embedding and hidden dimensions are 650.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pretrained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
        The pre-trained model achieves 89.01 ppl on wikitext-2
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab
    """
    predefined_args = {'embed_size': 650,
                       'hidden_size': 650,
                       'mode': 'lstm',
                       'num_layers': 2,
                       'tie_weights': True,
                       'dropout': 0.5}
    mutable_args = ['dropout']
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    return _get_rnn_model(StandardRNN, 'standard_lstm_lm_650', dataset_name, vocab, pretrained,
                          ctx, root, **predefined_args)


def standard_lstm_lm_1500(dataset_name=None, vocab=None, pretrained=False, ctx=cpu(),
                          root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Standard 2-layer LSTM language model with tied embedding and output weights.

    Both embedding and hidden dimensions are 1500.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pretrained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
        The pre-trained model is not provided yet.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab
    """
    predefined_args = {'embed_size': 1500,
                       'hidden_size': 1500,
                       'mode': 'lstm',
                       'num_layers': 2,
                       'tie_weights': True,
                       'dropout': 0.65}
    mutable_args = ['dropout']
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
           'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    return _get_rnn_model(StandardRNN, 'standard_lstm_lm_1500', dataset_name, vocab, pretrained,
                          ctx, root, **predefined_args)

def user_pretrained_lm(predefined_args=None, model_name=None, dataset_name=None, vocab=None, pretrained=False, ctx=cpu(),
                     root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""3-layer LSTM language model with weight-drop, variational dropout, and tied weights.

    Embedding size is 400, and hidden layer size is 1150.

    Parameters
    ----------
    dataset_name : str or None, default None
        The dataset name on which the pretrained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
        The pre-trained model must be specified by the user.
    vocab : gluonnlp.Vocab or None, default None
        Vocab object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab
    """
    if predefined_args.keys().__contains__('weight_drop') and predefined_args['weight_drop'] > 0:
        return _get_rnn_model(AWDRNN, model_name, dataset_name, vocab, pretrained,
                          ctx, root, **predefined_args)
    return _get_rnn_model(StandardRNN, model_name, dataset_name, vocab, pretrained,
                                    ctx, root, **predefined_args)

model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('d572ce7190e128b94aed403ad89a610aebe8a74a', 'standard_lstm_lm_1500_wikitext-2'),
        ('140416672f27691173523a7535b13cb3adf050a1', 'standard_lstm_lm_650_wikitext-2'),
        ('700b532dc96a29e39f45cb7dd632ce44e377a752', 'standard_lstm_lm_200_wikitext-2'),
        ('45d6df33f35715fb760ec8d18ed567016a897df7', 'awd_lstm_lm_1150_wikitext-2'),
        ('7894a046f8286db0d5d2ed672b60f4f52b4bc3aa', 'awd_lstm_lm_600_wikitext-2')
    ]})
