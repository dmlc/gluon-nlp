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

"""Question answering example."""
import argparse
import random
import mxnet as mx

from gluonnlp.data.squad_dataset import SQuAD
from scripts.question_answering.data_processing import preprocess_dataset

random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='MXNet Question Answering example on SQuAD dataset. '
                                             'We use Bi-Directional Attention Flow model.')

parser.add_argument('--question_max_length', type=int, default=30,
                    help='Maximum length of a question to use.')
parser.add_argument('--context_max_length', type=int, default=256,
                    help='Maximum length of a context to use')
parser.add_argument('--lr', type=float, default=2.5E-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=None,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                    help='report interval')
parser.add_argument('--save-prefix', type=str, default='qa-model',
                    help='path to save the final model')
parser.add_argument('--gpu', type=int, default=None,
                    help='id of the gpu to use. Set it to empty means to use cpu.')
args = parser.parse_args()


dataset = SQuAD(segment='train')
processed_dataset = preprocess_dataset(dataset, args.question_max_length, args.context_max_length)
