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
# pylint:disable=redefined-outer-name,logging-format-interpolation
"""PyTorch BERT parameter naming to Gluon BERT parameter naming.

Given a Gluon BERT model (eg. obtained with the convert_tf_gluon.py script) and
a pytorch_model.bin containing the same parameters, this script infers the
naming convention of PyTorch.

"""

import argparse
import json
import logging
import os
import sys

import gluonnlp as nlp
import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
from utils import load_text_vocab, tf_vocab_to_gluon_vocab

parser = argparse.ArgumentParser(description='Pytorch BERT Naming Convention',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='bert_12_768_12',
                    choices=['bert_12_768_12', 'bert_24_1024_16'], help='BERT model name')
parser.add_argument('--dataset_name', type=str, default='scibert_scivocab_uncased',
                    help='Dataset name')
parser.add_argument('--pytorch_checkpoint_dir', type=str,
                    help='Path to Tensorflow checkpoint folder.')
parser.add_argument('--debug', action='store_true', help='debugging mode')
parser.add_argument('--out', default='gluon_to_pytorch_naming.json',
                    help='Output file to store gluon to pytorch name mapping.')
args = parser.parse_args()
logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
logging.info(args)

# Load Gluon Model
bert, vocab = nlp.model.get_model(args.model, dataset_name=args.dataset_name, pretrained=True)
parameters = bert._collect_params_with_prefix()
parameters = {k: v.data().asnumpy() for k, v in parameters.items()}

# Load PyTorch Model
pytorch_parameters = torch.load(os.path.join(args.pytorch_checkpoint_dir, 'pytorch_model.bin'),
                                map_location=lambda storage, loc: storage)
pytorch_vocab = tf_vocab_to_gluon_vocab(
    load_text_vocab(os.path.join(args.pytorch_checkpoint_dir, 'vocab.txt')))
pytorch_parameters = {k: v.numpy() for k, v in pytorch_parameters.items()}

# Assert that vocabularies are equal
assert pytorch_vocab.idx_to_token == vocab.idx_to_token

mapping = dict()

for name, param in parameters.items():
    found_match = False
    for pytorch_name, pytorch_param in pytorch_parameters.items():
        if param.shape == pytorch_param.shape:
            if (param == pytorch_param).all():
                if found_match:
                    print('Found multiple matches for {}. '
                          'Ignoring new match {}'.format(name, pytorch_name))
                else:
                    found_match = True
                    mapping.update({name: pytorch_name})

        # We don't break here, in case there are mulitple matches

    if not found_match:
        raise RuntimeError('Pytorch and Gluon model do not match. '
                           'Cannot infer mapping of names.')

assert len(mapping) == len(parameters)

with open(args.out, 'w') as f:
    json.dump(mapping, f, indent="  ")
    print('Wrote mapping to {}'.format(args.out))
