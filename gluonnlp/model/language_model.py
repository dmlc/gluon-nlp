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
__all__ = ['AWDRNN', 'StandardRNN', 'awd_lstm_lm_1150', 'awd_lstm_lm_600',
           'standard_lstm_lm_200', 'standard_lstm_lm_650', 'standard_lstm_lm_1500']

import os
import warnings

from mxnet.gluon.model_zoo.model_store import get_model_file
from mxnet import nd, cpu, autograd
from mxnet.gluon.model_zoo import model_store

from gluonnlp.model import train
from gluonnlp.data.utils import _load_pretrained_vocab


class AWDRNN(train.AWDRNN):
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
    def __init__(self, mode, vocab_size, embed_size, hidden_size, num_layers,
                 tie_weights, dropout, weight_drop, drop_h,
                 drop_i, drop_e, **kwargs):
        super(AWDRNN, self).__init__(mode, vocab_size, embed_size, hidden_size, num_layers,
                                     tie_weights, dropout, weight_drop,
                                     drop_h, drop_i, drop_e, **kwargs)

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Implement forward computation.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(sequence_length, batch_size)`
            when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers.
            the initial state with shape `(1, batch_size, num_hidden)`

        Returns
        --------
        out: NDArray
            output tensor with shape `(sequence_length, batch_size, input_size)`
            when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers.
            the state with shape `(1, batch_size, num_hidden)`
        """
        encoded = self.embedding(inputs)
        if begin_state is None:
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

class StandardRNN(train.StandardRNN):
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
                 num_layers, dropout, tie_weights, **kwargs):
        if tie_weights:
            assert embed_size == hidden_size, 'Embedding dimension must be equal to ' \
                                              'hidden dimension in order to tie weights. ' \
                                              'Got: emb: {}, hid: {}.'.format(embed_size,
                                                                              hidden_size)
        super(StandardRNN, self).__init__(mode, vocab_size, embed_size, hidden_size,
                                          num_layers, dropout, tie_weights, **kwargs)

    def forward(self, inputs, begin_state=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.

        Parameters
        -----------
        inputs : NDArray
            input tensor with shape `(sequence_length, batch_size)`
              when `layout` is "TNC".
        begin_state : list
            initial recurrent state tensor with length equals to num_layers-1.
            the initial state with shape `(num_layers, batch_size, num_hidden)`

        Returns
        --------
        out: NDArray
            output tensor with shape `(sequence_length, batch_size, input_size)`
              when `layout` is "TNC".
        out_states: list
            output recurrent state tensor with length equals to num_layers-1.
            the state with shape `(num_layers, batch_size, num_hidden)`
        """
        encoded = self.embedding(inputs)
        if begin_state is None:
            begin_state = self.begin_state(batch_size=inputs.shape[1])
        encoded, state = self.encoder(encoded, begin_state)
        if self._dropout:
            encoded = nd.Dropout(encoded, p=self._dropout, axes=(0,))
        out = self.decoder(encoded)
        return out, state

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
        The pre-trained model achieves 73.32/69.74 ppl on Val and Test of wikitext-2 respectively.
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
        The pre-trained model achieves 84.61/80.96 ppl on Val and Test of wikitext-2 respectively.
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
        The pre-trained model achieves 108.25/102.26 ppl on Val and Test of wikitext-2 respectively.
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
        The pre-trained model achieves 98.96/93.90 ppl on Val and Test of wikitext-2 respectively.
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
        The pre-trained model achieves 98.29/92.83 ppl on Val and Test of wikitext-2 respectively.
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
    return _get_rnn_model(StandardRNN, 'standard_lstm_lm_1500',
                          dataset_name, vocab, pretrained, ctx, root, **predefined_args)

model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('d572ce7190e128b94aed403ad89a610aebe8a74a', 'standard_lstm_lm_1500_wikitext-2'),
        ('140416672f27691173523a7535b13cb3adf050a1', 'standard_lstm_lm_650_wikitext-2'),
        ('700b532dc96a29e39f45cb7dd632ce44e377a752', 'standard_lstm_lm_200_wikitext-2'),
        ('45d6df33f35715fb760ec8d18ed567016a897df7', 'awd_lstm_lm_1150_wikitext-2'),
        ('7894a046f8286db0d5d2ed672b60f4f52b4bc3aa', 'awd_lstm_lm_600_wikitext-2')
    ]})
