"""Transformer-XL Language Model
================================

This example shows how to build a Transformer-XL language model with Gluon NLP
Toolkit.

@article{dai2019transformer,
  title = {Transformer-XL: Attentive language models beyond a fixed-length context},
  author = {Dai, Zihang and Yang, Zhilin and Yang, Yiming and Cohen, William W
      and Carbonell, Jaime and Le, Quoc V and Salakhutdinov, Ruslan},
  journal = {arXiv preprint arXiv:1901.02860},
  year = {2019},
}

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
import itertools
import math
import sys
import time

import mxnet as mx

import gluonnlp as nlp


def evaluate(data_iter):
    """Evaluate the model on the dataset."""

    total_L = mx.nd.zeros(shape=(1, ))
    ntotal = 0

    mems = model.begin_mems(args.eval_batch_size, args.mem_len, context=ctx)
    for i, (data, target) in enumerate(data_iter):
        data = data.T.as_in_context(ctx)
        target = target.T.as_in_context(ctx)
        L, mems, _ = model(data, target, mems)  # Negative log likelihood of targets
        total_L += mx.nd.sum(L).as_in_context(mx.cpu())
        ntotal += L.size
        mx.nd.waitall()  # Avoid OOM due to pushing data too fast

        if i % args.log_every == 0:
            current_loss = total_L.asscalar() / ntotal
            print('Iter {} evaluation loss {:.2f}, ppl {:.2f}, bpc {:.2f}'.format(
                i, current_loss, math.exp(current_loss), current_loss / math.log(2)))

    return total_L.asscalar() / ntotal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer-XL Language Modeling.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wt103', 'text8', 'enwik8', 'lm1b'], help='Dataset name.')
    parser.add_argument('--split', type=str, default='test', choices=['valid', 'test'],
                        help='Which split to evaluate')
    parser.add_argument('--parameter-file', type=str, default=None, required=True,
                        help='File storing pre-trained parameters for the model.')
    parser.add_argument('--vocab-file', type=str, default=None, required=True,
                        help='File storing nlp.Vocab corresponding to --parameter-file.')

    parser.add_argument('--mem-len', type=int, default=1600,
                        help='length of the retained previous heads')
    parser.add_argument('--bptt', type=int, default=128,
                        help='The number of tokens per batch dimension per sample.')
    parser.add_argument('--clamp-len', type=int, default=1000,
                        help='max positional embedding index')

    parser.add_argument('--log-every', type=int, default=10,
                        help='Log every `--log-every` iterations.')

    # TODO: training not yet supported
    parser.add_argument('--eval-only', action='store_true', required=True,
                        help='Only evaluate the trained model')
    parser.add_argument('--eval-batch-size', type=int, default=64,
                        help='Batch size for evaluation.')
    parser.add_argument('--gpu', type=int, help='GPU id')
    args = parser.parse_args()

    start_time = time.time()

    # Model
    from transformer.model import get_model
    with open(args.vocab_file, 'r') as f:
        vocab = nlp.Vocab.from_json(f.read())

    ctx = mx.gpu(args.gpu) if args.gpu is not None else mx.cpu()
    model, vocab = get_model('transformerxl', vocab=vocab, dataset_name=args.dataset,
                             clamp_len=args.clamp_len)
    model.initialize(ctx=ctx)
    model.load_parameters(args.parameter_file, ignore_extra=False)
    model.hybridize()
    print(model)

    # Data
    if args.dataset == 'wt103':
        val_dataset, test_dataset = [
            nlp.data.WikiText103(segment=segment, skip_empty=False, bos=vocab.bos_token,
                                 eos=vocab.eos_token) for segment in ['val', 'test']
        ]
    elif args.dataset == 'lm1b':
        # bos=vocab.eos_token is not a typo: tf uses ['<S>'] + symbols + ['<S>']
        test_datasets = list(
            nlp.data.GBWStream(segment='test', skip_empty=True, bos=vocab.eos_token,
                               eos=vocab.eos_token))
        assert len(test_datasets) == 1
        test_dataset = mx.gluon.data.SimpleDataset(
            list(itertools.chain.from_iterable(test_datasets[0])))
        val_dataset = None
    elif args.dataset == 'text8':
        dataset = nlp.data.Text8(max_sentence_length=None)
        chars = list(itertools.chain.from_iterable(list(w) + ['_'] for w in dataset[0]))
        num_test_chars = 5000000
        val_dataset = mx.gluon.data.SimpleDataset(chars[-2 * num_test_chars:-num_test_chars])
        test_dataset = mx.gluon.data.SimpleDataset(chars[-num_test_chars:])
    elif args.dataset == 'enwik8':
        val_dataset, test_dataset = [
            mx.gluon.data.SimpleDataset(
                list(itertools.chain.from_iterable(nlp.data.Enwik8(segment=segment))))
            for segment in ['val', 'test']
        ]
    else:
        print('Dataset unsupported by this script.')
        sys.exit(1)

    eval_batchify = nlp.data.batchify.CorpusBPTTBatchify(vocab, args.bptt, args.eval_batch_size,
                                                         last_batch='discard')

    # Evaluate
    test_loss = None
    valid_loss = None
    if args.split in ('valid', 'all') and val_dataset is not None:
        val_data = eval_batchify(val_dataset)
        valid_loss = evaluate(val_data)
    if args.split in ('test', 'all') and test_dataset is not None:
        test_data = eval_batchify(test_dataset)
        test_loss = evaluate(test_data)

    if test_loss is not None:
        print('Best test loss {:.2f}, test ppl {:.2f}, test bpc {:.2f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
    if valid_loss is not None:
        print('Best validation loss {:.2f}, val ppl {:.2f}, val bpc {:.2f}'.format(
            valid_loss, math.exp(valid_loss), valid_loss / math.log(2)))

    print('Total time cost {:.2f}s'.format(time.time() - start_time))
