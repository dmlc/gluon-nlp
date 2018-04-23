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
"""Machine translation models and translators."""


__all__ = ['NMTModel', 'BeamSearchTranslator']

import warnings
import numpy as np
from mxnet.gluon import Block
from mxnet.gluon import nn
import mxnet as mx
from .beam_search import BeamSearchScorer, BeamSearchSampler


class NMTModel(Block):
    """Model for Neural Machine Translation.

    Parameters
    ----------
    src_vocab : Vocab
        Source vocabulary.
    tgt_vocab : Vocab
        Target vocabulary.
    encoder : Seq2SeqEncoder
        Encoder that encodes the input sentence.
    decoder : Seq2SeqDecoder
        Decoder that generates the predictions based on the output of the encoder.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the source and target embeddings
        if src_embed and tgt_embed are None.
    embed_dropout : float, default 0.0
        Dropout rate of the embedding weights. It is used to generate the source and target
        embeddings if src_embed and tgt_embed are None.
    embed_initializer : Initializer, default mx.init.Uniform(0.1)
        Initializer of the embedding weights. It is used to generate ghe source and target
        embeddings if src_embed and tgt_embed are None.
    src_embed : Block or None, default None
        The source embedding. If set to None, src_embed will be constructed using embed_size and
        embed_dropout.
    tgt_embed : Block or None, default None
        The target embedding. If set to None and the tgt_embed will be constructed using
        embed_size and embed_dropout. Also if `share_embed` is turned on, we will set tgt_embed
        to be the same as src_embed.
    share_embed : bool, default False
        Whether to share the src/tgt embeddings or not.
    tgt_proj : Block or None, default None
        Layer that projects the decoder outputs to the target vocabulary.
    prefix : str or None
        See document of `Block`.
    params : ParameterDict or None
        See document of `Block`.
    """
    def __init__(self, src_vocab, tgt_vocab, encoder, decoder,
                 embed_size=None, embed_dropout=0.0, embed_initializer=mx.init.Uniform(0.1),
                 src_embed=None, tgt_embed=None, share_embed=False, tgt_proj=None,
                 prefix=None, params=None):
        super(NMTModel, self).__init__(prefix=prefix, params=params)
        self.tgt_vocab = tgt_vocab
        self.src_vocab = src_vocab
        self.encoder = encoder
        self.decoder = decoder
        self._shared_embed = share_embed
        if embed_dropout is None:
            embed_dropout = 0.0
        # Construct src embedding
        if share_embed and tgt_embed is not None:
            warnings.warn('"share_embed" is turned on and \"tgt_embed\" is not None. '
                          'In this case, the provided "tgt_embed" will be overwritten by the '
                          '"src_embed". Is this intended?')
        if src_embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "src_embed" is not ' \
                                           'given.'
            with self.name_scope():
                self.src_embed = nn.HybridSequential(prefix='src_embed_')
                with self.src_embed.name_scope():
                    self.src_embed.add(nn.Embedding(input_dim=len(src_vocab), output_dim=embed_size,
                                                    weight_initializer=embed_initializer))
                    self.src_embed.add(nn.Dropout(rate=embed_dropout))
        else:
            self.src_embed = src_embed
        # Construct tgt embedding
        if share_embed:
            self.tgt_embed = self.src_embed
        else:
            if tgt_embed is not None:
                self.tgt_embed = tgt_embed
            else:
                assert embed_size is not None,\
                    '"embed_size" cannot be None if "tgt_embed" is ' \
                    'not given and "shared_embed" is not turned on.'
                with self.name_scope():
                    self.tgt_embed = nn.HybridSequential(prefix='tgt_embed_')
                    with self.tgt_embed.name_scope():
                        self.tgt_embed.add(
                            nn.Embedding(input_dim=len(tgt_vocab), output_dim=embed_size,
                                         weight_initializer=embed_initializer))
                        self.tgt_embed.add(nn.Dropout(rate=embed_dropout))
        # Construct tgt proj
        if tgt_proj is None:
            with self.name_scope():
                self.tgt_proj = nn.Dense(units=len(tgt_vocab), flatten=False, prefix='tgt_proj_')
        else:
            self.tgt_proj = tgt_proj

    def encode(self, inputs, states=None, valid_length=None):
        """Encode the input sequence.

        Parameters
        ----------
        inputs : NDArray
        states : list of NDArrays or None, default None
        valid_length : NDArray or None, default None

        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        return self.encoder(self.src_embed(inputs), states, valid_length)

    def decode_seq(self, inputs, states, valid_length=None):
        """Decode given the input sequence.

        Parameters
        ----------
        inputs : NDArray
        states : list of NDArrays
        valid_length : NDArray or None, default None

        Returns
        -------
        output : NDArray
            The output of the decoder. Shape is (batch_size, length, tgt_word_num)
        states: list
            The new states of the decoder
        additional_outputs : list
            Additional outputs of the decoder, e.g, the attention weights
        """
        outputs, states, additional_outputs =\
            self.decoder.decode_seq(inputs=self.tgt_embed(inputs),
                                    states=states,
                                    valid_length=valid_length)
        outputs = self.tgt_proj(outputs)
        return outputs, states, additional_outputs

    def decode_step(self, step_input, states):
        """One step decoding of the translation model.

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size,)
        states : list of NDArrays

        Returns
        -------
        step_output : NDArray
            Shape (batch_size, C_out)
        states : list
        step_additional_outputs : list
            Additional outputs of the step, e.g, the attention weights
        """
        step_output, states, step_additional_outputs =\
            self.decoder(self.tgt_embed(step_input), states)
        step_output = self.tgt_proj(step_output)
        return step_output, states, step_additional_outputs

    def __call__(self, src_seq, tgt_seq, src_valid_length=None, tgt_valid_length=None):  #pylint: disable=arguments-differ
        """Generate the prediction given the src_seq and tgt_seq.

        This is used in training an NMT model.

        Parameters
        ----------
        src_seq : NDArray
        tgt_seq : NDArray
        src_valid_length : NDArray or None
        tgt_valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, tgt_length, tgt_word_num)
        additional_outputs : list
            Additional outputs, e.g, the attention weights
        """
        return self.forward(src_seq, tgt_seq, src_valid_length, tgt_valid_length)

    def forward(self, src_seq, tgt_seq, src_valid_length=None, tgt_valid_length=None):  #pylint: disable=arguments-differ
        encoder_outputs = self.encode(src_seq, valid_length=src_valid_length)
        decoder_states = self.decoder.init_state_from_encoder(encoder_outputs,
                                                              encoder_valid_length=src_valid_length)
        outputs, _, additional_outputs =\
            self.decode_seq(tgt_seq, decoder_states, tgt_valid_length)
        return outputs, additional_outputs


class BeamSearchTranslator(object):
    """Beam Search Translator

    Parameters
    ----------
    model : NMTModel
        The neural machine translation model
    beam_size : int
        Size of the beam
    scorer : BeamSearchScorer
        Score function used in beamsearch
    max_length : int
        The maximum decoding length
    """
    def __init__(self, model, beam_size=1, scorer=BeamSearchScorer(), max_length=100):
        self._model = model
        self._sampler = BeamSearchSampler(
            decoder=self._decode_logprob,
            beam_size=beam_size,
            eos_id=model.tgt_vocab.token_to_idx[model.tgt_vocab.eos_token],
            scorer=scorer,
            max_length=max_length)

    def _decode_logprob(self, step_input, states):
        out, states, _ = self._model.decode_step(step_input, states)
        return mx.nd.log_softmax(out), states

    def translate(self, src_seq, src_valid_length):
        """Get the translation result given the input sentence.

        Parameters
        ----------
        src_seq : mx.nd.NDArray
            Shape (batch_size, length)
        src_valid_length : mx.nd.NDArray
            Shape (batch_size,)

        Returns
        -------
        samples : NDArray
            Samples draw by beam search. Shape (batch_size, beam_size, length). dtype is int32.
        scores : NDArray
            Scores of the samples. Shape (batch_size, beam_size). We make sure that scores[i, :] are
            in descending order.
        valid_length : NDArray
            The valid length of the samples. Shape (batch_size, beam_size). dtype will be int32.
        """
        batch_size = src_seq.shape[0]
        encoder_outputs = self._model.encode(src_seq, valid_length=src_valid_length)
        decoder_states = self._model.decoder.init_state_from_encoder(encoder_outputs,
                                                                     src_valid_length)
        inputs = mx.nd.full(shape=(batch_size,), ctx=src_seq.context, dtype=np.float32,
                            val=self._model.tgt_vocab.token_to_idx[self._model.tgt_vocab.bos_token])
        samples, scores, sample_valid_length = self._sampler(inputs, decoder_states)
        return samples, scores, sample_valid_length
