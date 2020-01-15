# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Script for model comparison between TF and Gluon."""

import argparse
import logging
import os
import sys

import mxnet as mx
import numpy as np
import torch

import gluonnlp as nlp
import transformers


def compare_xlnet(args):
    batch_size, qlen, mlen = 2, 16, 100

    model_p = transformers.XLNetLMHeadModel.from_pretrained(
        'xlnet-base-cased'
        if args.model_name == 'xlnet_cased_L-12_H-768_A-12' else 'xlnet-large-cased', dropout=0)
    model_p.transformer.attentions = False  # no change of default
    model_p.transformer.output_hidden_states = True
    model_p.transformer.mem_len = mlen

    if args.model_name == 'xlnet_cased_L-12_H-768_A-12':
        kwargs = {
            'hidden_size': 3072,
            'units': 768,
            'activation': 'approx_gelu',
            'num_heads': 12,
            'num_layers': 12,
            'vocab_size': 32000
        }
    elif args.model_name == 'xlnet_cased_L-24_H-1024_A-16':
        kwargs = {
            'hidden_size': 4096,
            'units': 1024,
            'activation': 'approx_gelu',
            'num_heads': 16,
            'num_layers': 24,
            'vocab_size': 32000
        }

    with open(args.gluon_vocab_file, 'r') as f:
        vocab = nlp.Vocab.from_json(f.read())
    ctx = mx.cpu()
    assert kwargs['vocab_size'] == len(vocab)
    clamp_len = model_p.transformer.clamp_len if model_p.transformer.clamp_len > 0 else None
    model = XLNet(clamp_len=clamp_len, **kwargs)
    model.initialize(ctx=ctx)
    model.load_parameters(args.gluon_parameter_file, ignore_extra=False)
    model.hybridize()

    # Computation
    mems = model.begin_mems(batch_size, mlen, context=mx.cpu())
    x = mx.nd.ones(shape=(batch_size, qlen))
    token_types = mx.nd.ones(shape=(batch_size, qlen))
    output, new_mems = model(x, token_types, mems)

    x_p = torch.tensor(x.asnumpy(), dtype=torch.long)
    mems_p = [torch.tensor(mems_i.transpose((1, 0, 2)).asnumpy()) for mems_i in mems]
    token_types_p = torch.tensor(token_types.asnumpy(), dtype=torch.long)
    output_p, new_mems_p, hids_p = model_p(x_p, token_type_ids=token_types_p, mems=mems_p)

    for i in range(kwargs['num_layers']):
        a, b = new_mems[i][:, -qlen:].asnumpy(), hids_p[i].detach().numpy()
        assert np.all(np.isclose(a, b, atol=1e-5))
    assert np.all(np.isclose(output.asnumpy(), output_p.detach().numpy(), atol=5e-5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparison script for Tensorflow and GLuon XLNet model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-name', type=str, required=True,
                        choices=['xlnet_cased_L-12_H-768_A-12',
                                 'xlnet_cased_L-24_H-1024_A-16'], help='Model name')
    parser.add_argument('--gluon-parameter-file', type=str, required=True,
                        help='gluon parameter file name.')
    parser.add_argument('--gluon-vocab-file', type=str, required=True,
                        help='gluon vocab file corresponding to --gluon_parameter_file.')
    parser.add_argument('--debug', action='store_true', help='debugging mode')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.info(args)
    sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
    from transformer import XLNet

    compare_xlnet(args)
