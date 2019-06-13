"""
Generate Sentences by Sampling and Beam Search
==============================================

This example shows how to load a pre-trained language model on wikitext-2 in Gluon NLP Toolkit model
zoo, and use sequence sampler and beam search sampler on the language model to generate sentences.
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
# pylint:disable=missing-docstring
import argparse

import numpy as np
import mxnet as mx
import gluonnlp as nlp

import model # local 'model' module with the addition of GPT-2


parser = argparse.ArgumentParser(description='Generate sentences by beam search. '
                                             'We load a LSTM model that is pre-trained on '
                                             'WikiText as our encoder.')

# beam search sampler options
subparsers = parser.add_subparsers(help='Sequence generation methods.',
                                   dest='command')
subparsers.required = True
beam_search_parser = subparsers.add_parser('beam-search', help='Use beam search for decoding.')
beam_search_parser.add_argument('--alpha', type=float, default=0.0,
                                help='Alpha in the length penalty term.')
beam_search_parser.add_argument('--k', type=int, default=5, help='K in the length penalty term.')

# random sampler options
random_sample_parser = subparsers.add_parser('random-sample',
                                             help='Use random sampling for decoding.')
random_sample_parser.add_argument('--temperature', type=float, default=1.0,
                                  help='Softmax temperature used in sampling.')
random_sample_parser.add_argument('--use-top-k', type=int, required=False,
                                  help='Sample only from the top-k candidates.')

# shared options
for p in [beam_search_parser, random_sample_parser]:
    p.add_argument('--gpu', type=int, default=0,
                   help='id of the gpu to use. Set it to empty means to use cpu.')
    p.add_argument('--lm-model', type=str, default='awd_lstm_lm_1150',
                   help='type of the pre-trained model to load, can be "standard_lstm_lm_200", '
                        '"standard_lstm_lm_650", "standard_lstm_lm_1500", '
                        '"awd_lstm_lm_1150", etc.')
    p.add_argument('--max-length', type=int, default=20, help='Maximum sentence length.')
    p.add_argument('--print-num', type=int, default=3, help='Number of sentences to display.')
    p.add_argument('--bos', type=str, default='I think this works')
    p.add_argument('--beam-size', type=int, default=5,
                   help='Beam size in the beam search sampler.')

args = parser.parse_args()

print(args)
if args.gpu is not None and args.gpu < mx.context.num_gpus():
    ctx = mx.gpu(args.gpu)
else:
    if args.gpu:
        print('Specified GPU id {} does not exist. Available #GPUs: {}. Using CPU instead.'\
                .format(args.gpu, mx.context.num_gpus()))
    ctx = mx.cpu()

assert 0 < args.print_num <= args.beam_size,\
    'print_num must be between {} and {}, received={}'.format(1, args.beam_size, args.print_num)


# Define the decoder function, we use log_softmax to map the output scores to log-likelihoods
# Also, we transform the layout to NTC
class LMDecoder(object):
    def __init__(self, net):
        self.net = net

    def __call__(self, inputs, states):
        outputs, states = self.net(mx.nd.expand_dims(inputs, axis=0), states)
        return outputs[0], states

    def state_info(self, *arg, **kwargs):
        return self.net.state_info(*arg, **kwargs)

class GPT2Decoder(LMDecoder):
    def __call__(self, inputs, states):
        inputs = mx.nd.expand_dims(inputs, axis=1)
        out, new_states = self.net(inputs, states)
        out = mx.nd.slice_axis(out, axis=1, begin=0, end=1).reshape((inputs.shape[0], -1))
        return out, new_states

def get_decoder_vocab(lm_model):
    if lm_model.startswith('gpt2'):
        dataset_name = 'openai_webtext'
        decoder_cls = GPT2Decoder
    else:
        dataset_name = 'wikitext-2'
        decoder_cls = LMDecoder
    lm_model, vocab = model.get_model(name=lm_model,
                                      dataset_name=dataset_name,
                                      pretrained=True,
                                      ctx=ctx)
    decoder = decoder_cls(lm_model)
    return decoder, vocab

def get_tokenizer(lm_model):
    if lm_model.startswith('gpt2'):
        return nlp.data.GPT2BPETokenizer(), nlp.data.GPT2BPEDetokenizer()
    else:
        return nlp.data.SacreMosesTokenizer(), nlp.data.SacreMosesDetokenizer(return_str=True)

def get_initial_input_state(decoder, bos_ids):
    if isinstance(decoder, GPT2Decoder):
        inputs, begin_states = decoder.net(
            mx.nd.array([bos_ids], dtype=np.int32, ctx=ctx), None)
        inputs = inputs[:, -1, :]
        smoothed_probs = (inputs / args.temperature).softmax(axis=1)
        inputs = mx.nd.sample_multinomial(smoothed_probs, dtype=np.int32)
        return inputs, begin_states
    else:
        begin_states = decoder.net.begin_state(batch_size=1, ctx=ctx)
        if len(bos_ids) > 1:
            _, begin_states = decoder.net(mx.nd.expand_dims(mx.nd.array(bos_ids[:-1], ctx=ctx),
                                                            axis=1),
                                          begin_states)
        inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_ids[-1])
        return inputs, begin_states


def generate():
    assert not args.lm_model.startswith('gpt2') or args.command != 'beam-search'
    decoder, vocab = get_decoder_vocab(args.lm_model)
    tokenizer, detokenizer = get_tokenizer(args.lm_model)
    bos_str = args.bos
    if not bos_str.startswith(' '):
        bos_str = ' ' + bos_str
    bos_tokens = tokenizer(bos_str)
    bos_ids = vocab[bos_tokens]
    eos_id = vocab[vocab.eos_token]
    if args.command == 'random-sample':
        print('Sampling Parameters: beam_size={}, temperature={}, use_top_k={}'\
                .format(args.beam_size, args.temperature, args.use_top_k))
        sampler = nlp.model.SequenceSampler(beam_size=args.beam_size,
                                            decoder=decoder,
                                            eos_id=eos_id,
                                            max_length=args.max_length - len(bos_tokens),
                                            temperature=args.temperature,
                                            top_k=args.use_top_k)
    else:
        print('Beam Seach Parameters: beam_size={}, alpha={}, K={}'\
                .format(args.beam_size, args.alpha, args.k))
        scorer = nlp.model.BeamSearchScorer(alpha=args.alpha, K=args.k, from_logits=False)
        sampler = nlp.model.BeamSearchSampler(beam_size=args.beam_size,
                                              decoder=decoder,
                                              eos_id=eos_id,
                                              scorer=scorer,
                                              max_length=args.max_length - len(bos_tokens))
    inputs, begin_states = get_initial_input_state(decoder, bos_ids)
    # samples have shape (1, beam_size, length), scores have shape (1, beam_size)
    samples, scores, valid_lengths = sampler(inputs, begin_states)
    samples = samples[0].asnumpy()
    scores = scores[0].asnumpy()
    valid_lengths = valid_lengths[0].asnumpy()

    print('Generation Result:')
    for i in range(args.print_num):
        generated_tokens = [vocab.idx_to_token[ele] for ele in samples[i][:valid_lengths[i]]]
        tokens = bos_tokens + generated_tokens[1:]
        print([detokenizer(tokens).strip(), scores[i]])


if __name__ == '__main__':
    generate()
