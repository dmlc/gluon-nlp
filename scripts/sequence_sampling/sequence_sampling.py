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
import mxnet as mx
import gluonnlp as nlp


parser = argparse.ArgumentParser(description='Generate sentences by beam search. '
                                             'We load a LSTM model that is pre-trained on '
                                             'WikiText as our encoder.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--use-sampling', action='store_true',
                   help='Use sampling instead of beam search.')
group.add_argument('--use-beam-search', action='store_true',
                   help='Use beam search instead of random sampling.')
parser.add_argument('--lm_model', type=str, default='awd_lstm_lm_1150',
                    help='type of the pre-trained model to load, can be "standard_lstm_lm_200", '
                         '"standard_lstm_lm_650", "standard_lstm_lm_1500", '
                         '"awd_lstm_lm_1150", etc.')
parser.add_argument('--beam_size', type=int, default=5,
                    help='Beam size in the beam search sampler.')
parser.add_argument('--alpha', type=float, default=0.0, help='Alpha in the length penalty term.')
parser.add_argument('--k', type=int, default=5, help='K in the length penalty term.')
parser.add_argument('--bos', type=str, default='I', nargs='+')
parser.add_argument('--eos', type=str, default='.')
parser.add_argument('--max_length', type=int, default=20, help='Maximum sentence length.')
parser.add_argument('--print_num', type=int, default=3, help='Number of sentences to display.')
parser.add_argument('--gpu', type=int, default=None,
                    help='id of the gpu to use. Set it to empty means to use cpu.')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='Softmax temperature used in sampling.')
args = parser.parse_args()
if not args.use_beam_search and not args.use_sampling:
    parser.error('No decoding option specified. Add one of --use-sampling and --use-beam-search.')

print(args)
if args.gpu is None:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(args.gpu)

if isinstance(args.bos, str):
    args.bos = [args.bos]

lm_model, vocab = nlp.model.get_model(name=args.lm_model,
                                      dataset_name='wikitext-2',
                                      pretrained=True,
                                      ctx=ctx)

assert 0 < args.print_num <= args.beam_size,\
    'print_num must be between {} and {}, received={}'.format(1, args.beam_size, args.print_num)


def _transform_layout(data):
    if isinstance(data, list):
        return [_transform_layout(ele) for ele in data]
    elif isinstance(data, mx.nd.NDArray):
        return mx.nd.transpose(data, axes=(1, 0, 2))
    else:
        raise NotImplementedError


# Define the decoder function, we use log_softmax to map the output scores to log-likelihoods
# Also, we transform the layout to NTC
class LMDecoder(object):
    def __init__(self, model):
        self._model = model

    def __call__(self, inputs, states):
        outputs, states = self._model(mx.nd.expand_dims(inputs, axis=0), states)
        return outputs[0], states

    def state_info(self, *arg, **kwargs):
        return self._model.state_info(*arg, **kwargs)


def generate():
    bos_ids = [vocab[ele] for ele in args.bos]
    eos_id = vocab[args.eos]
    begin_states = lm_model.begin_state(batch_size=1, ctx=ctx)
    if len(bos_ids) > 1:
        _, begin_states = lm_model(mx.nd.expand_dims(mx.nd.array(bos_ids[:-1], ctx=ctx), axis=1),
                                   begin_states)
    inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_ids[-1])
    decoder = LMDecoder(lm_model)
    if args.use_sampling:
        sampler = nlp.model.SequenceSampler(beam_size=args.beam_size,
                                            decoder=decoder,
                                            eos_id=eos_id,
                                            max_length=args.max_length,
                                            temperature=args.temperature)
    else:
        scorer = nlp.model.BeamSearchScorer(alpha=args.alpha, K=args.k, from_logits=False)
        sampler = nlp.model.BeamSearchSampler(beam_size=args.beam_size,
                                              decoder=decoder,
                                              eos_id=eos_id,
                                              scorer=scorer,
                                              max_length=args.max_length)
    # samples have shape (1, beam_size, length), scores have shape (1, beam_size)
    samples, scores, valid_lengths = sampler(inputs, begin_states)
    samples = samples[0].asnumpy()
    scores = scores[0].asnumpy()
    valid_lengths = valid_lengths[0].asnumpy()
    if not args.use_sampling:
        print('Beam Seach Parameters: beam_size={}, alpha={}, K={}'.format(args.beam_size,
                                                                           args.alpha,
                                                                           args.k))
    else:
        print('Sampling Parameters: beam_size={}, temperature={}'.format(args.beam_size,
                                                                         args.temperature))

    print('Generation Result:')
    for i in range(args.print_num):
        sentence = args.bos[:-1] +\
                   [vocab.idx_to_token[ele] for ele in samples[i][:valid_lengths[i]]]
        print([' '.join(sentence), scores[i]])


if __name__ == '__main__':
    generate()
