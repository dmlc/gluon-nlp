"""
Generate Sentences by Beam Search
==================================

This example shows how to load a pretrained language model on wikitext-2 in Gluon NLP Toolkit model
zoo, and use the language model encoder to generate sentences.
"""

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

import argparse
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp


parser = argparse.ArgumentParser(description='Generate sentences by beam search. '
                                             'We load a LSTM model that is pretrained on '
                                             'WikiText as our encoder.')
parser.add_argument('--lm_model', type=str, default='awd_lstm_lm_1150',
                    help='type of the pretrained model to load, can be "standard_lstm_lm_200", '
                         '"standard_lstm_lm_650", "standard_lstm_lm_1500", '
                         '"awd_lstm_lm_1150", etc.')
parser.add_argument('--beam_size', type=int, default=4,
                    help='Beam size in the beam search sampler.')
parser.add_argument('--alpha', type=int, default=0.0, help='Alpha in the length penalty term.')
parser.add_argument('--k', type=int, default=5, help='K in the length penalty term.')
parser.add_argument('--bos', type=str, default='It')
parser.add_argument('--eos', type=str, default='.')
parser.add_argument('--max_length', type=int, default=20, help='Maximum sentence length.')
parser.add_argument('--gpu', type=int, default=None,
                    help='id of the gpu to use. Set it to empty means to use cpu.')
args = parser.parse_args()
print(args)
if args.gpu is None:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(args.gpu)


def generate():
    lm_model, vocab = nlp.model.get_model(name=args.lm_model,
                                          dataset_name='wikitext-2',
                                          pretrained=True,
                                          ctx=ctx)
    # Define the decoder function, we use log_softmax to map the output scores to log-likelihoods
    decoder = lambda inputs, states: mx.nd.log_softmax(lm_model(mx.nd.expand_dims(inputs, axis=0),
                                                                states))
    # Get the bos_id and eos_id based on the vocabulary
    bos_id = vocab[args.bos]
    eos_id = vocab[args.eos]
    begin_states = lm_model.begin_state(batch_size=1, ctx=ctx)
    inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_id)
    scorer = nlp.model.BeamSearchScorer(alpha=args.alpha, K=args.k)
    sampler = nlp.model.BeamSearchSampler(beam_size=args.beam_size,
                                          decoder=decoder,
                                          eos_id=eos_id,
                                          scorer=scorer,
                                          max_length=args.max_length)
    samples, scores, valid_lengths = sampler(inputs, begin_states)
    samples = samples[0].asnumpy()
    scores = scores[0].asnumpy()
    valid_lengths = valid_lengths[0].asnumpy()
    print("Generation Result:")
    for i in range(args.beam_size):
        sentence = [vocab.idx_to_token[ele] for ele in samples[i][:valid_lengths[i]]]
        print(tuple(' '.join(sentence), scores[i]))


if __name__ == '__main__':
    generate()

