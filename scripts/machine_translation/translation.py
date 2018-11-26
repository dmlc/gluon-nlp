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


__all__ = ['BeamSearchTranslator']

import numpy as np
import mxnet as mx
from gluonnlp.model import BeamSearchScorer, BeamSearchSampler

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
        encoder_outputs, _ = self._model.encode(src_seq, valid_length=src_valid_length)
        decoder_states = self._model.decoder.init_state_from_encoder(encoder_outputs,
                                                                     src_valid_length)
        inputs = mx.nd.full(shape=(batch_size,), ctx=src_seq.context, dtype=np.float32,
                            val=self._model.tgt_vocab.token_to_idx[self._model.tgt_vocab.bos_token])
        samples, scores, sample_valid_length = self._sampler(inputs, decoder_states)
        return samples, scores, sample_valid_length
