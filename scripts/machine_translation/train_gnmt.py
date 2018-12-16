"""
Google Neural Machine Translation
=================================

This example shows how to implement the GNMT model with Gluon NLP Toolkit.

@article{wu2016google,
  title={Google's neural machine translation system:
   Bridging the gap between human and machine translation},
  author={Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V and
   Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and
   Macherey, Klaus and others},
  journal={arXiv preprint arXiv:1609.08144},
  year={2016}
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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import time
import random
import os
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon, ndarray
import gluonnlp as nlp

from gluonnlp.model.translation import NMTModel
from gnmt import get_gnmt_encoder_decoder
from translation import BeamSearchTranslator
from loss import SoftmaxCEMaskedLoss
from utils import logging_config
from bleu import compute_bleu
import dataprocessor

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Google NMT model')
parser.add_argument('--dataset', type=str, default='IWSLT2015',
                    help='Dataset to use, default is IWSLT2015.')
parser.add_argument('--src_lang', type=str, default='en', help='Source language, '
                                                               'default is en.')
parser.add_argument('--tgt_lang', type=str, default='vi', help='Target language, '
                                                               'default is vi.')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit, , '
                                                           'default is 40.')
parser.add_argument('--num_hidden', type=int, default=128,
                    help='Dimension of the rnn hidden states. default is 128.')
parser.add_argument('--num_embedding', type=int, default=128,
                    help='Dimension of the embedding, default is 128. vectors and states.')
parser.add_argument('--attention_type', type=str, default='scaled_luong',
                    help=' Attention type, default \'is scaled_luong\'.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout), default is 0.2.')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the encoder and decoder, default is 2.')
parser.add_argument('--num_bi_layers', type=int, default=1,
                    help='number of bidirectional layers in the encoder and decoder, '
                         'default is 1.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size, default is 128.')
parser.add_argument('--beam_size', type=int, default=4, help='Beam size, default is 4.')
parser.add_argument('--lp_alpha', type=float, default=1.0,
                    help='Alpha used in calculating the length penalty, default is 1.0.')
parser.add_argument('--lp_k', type=int, default=5,
                    help='K used in calculating the length penalty, default is 5.')
parser.add_argument('--test_batch_size', type=int, default=32,
                    help='Test batch size, default is 32.')
parser.add_argument('--num_buckets', type=int, default=5,
                    help='Bucket number, default is 5.')
parser.add_argument('--bucket_scheme', type=str, default='constant',
                    help='Strategy for generating bucket keys. It supports: '
                         '"constant": all the buckets have the same width; '
                         '"linear": the width of bucket increases linearly; '
                         '"exp": the width of bucket increases exponentially'
                         'default is "constant".')
parser.add_argument('--bucket_ratio', type=float, default=0.0,
                    help='Ratio for increasing the throughput of the bucketing '
                         'default is 0.0.')
parser.add_argument('--src_max_len', type=int, default=50,
                    help='Maximum length of the source sentence, default is 50.')
parser.add_argument('--tgt_max_len', type=int, default=50,
                    help='Maximum length of the target sentence, default is 50.')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimization algorithm, default is adam.')
parser.add_argument('--lr', type=float, default=1E-3,
                    help='Initial learning rate, default is 1E-3.')
parser.add_argument('--lr_update_factor', type=float, default=0.5,
                    help='Learning rate decay factor, default is 0.5.')
parser.add_argument('--clip', type=float, default=5.0,
                    help='gradient clipping, default is 5.0.')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval, default is 100.')
parser.add_argument('--save_dir', type=str, default='out_dir',
                    help='directory path to save the final model and training log, '
                         'default is \'out_dir\'.')
parser.add_argument('--gpus', type=str, default=None,
                    help='id of the gpus to use, eg:0,1,2,3 means use 4 GPUs. '
                         'Set it to empty means to use cpu.')

args = parser.parse_args()
print(args)
logging_config(args.save_dir)


data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab\
    = dataprocessor.load_translation_data(dataset=args.dataset, bleu='tweaked', args=args)

dataprocessor.write_sentences(val_tgt_sentences, os.path.join(args.save_dir, 'val_gt.txt'))
dataprocessor.write_sentences(test_tgt_sentences, os.path.join(args.save_dir, 'test_gt.txt'))

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                     for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                      for i, ele in enumerate(data_test)])

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
    [mx.gpu(int(x)) for x in args.gpus.split(',')]
ctx = context[0]

# load GNMT model
encoder, decoder = get_gnmt_encoder_decoder(hidden_size=args.num_hidden,
                                            dropout=args.dropout,
                                            num_layers=args.num_layers,
                                            num_bi_layers=args.num_bi_layers,
                                            attention_cell=args.attention_type)

model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 embed_size=args.num_embedding, prefix='gnmt_')


model.initialize(init=mx.init.Uniform(0.1), ctx=context)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

translator = BeamSearchTranslator(model=model, beam_size=args.beam_size,
                                  scorer=nlp.model.BeamSearchScorer(alpha=args.lp_alpha,
                                                                    K=args.lp_k),
                                  max_length=args.tgt_max_len + 100)
logging.info('Use beam_size={}, alpha={}, K={}'.format(args.beam_size, args.lp_alpha, args.lp_k))


loss_function = SoftmaxCEMaskedLoss()
loss_function.hybridize(static_alloc=static_alloc)


def evaluate(data_loader):
    """Evaluate given the data loader
    Parameters
    ----------
    data_loader : DataLoader
    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0
    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(data_loader):
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        # Translate
        samples, _, sample_valid_length =\
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence
    return avg_loss, real_translation_out


def clip_global_norm(arrays, max_norm, context=[mx.cpu()]):
    """Enable the clip_global_norm function at multi contexts.
       Reference to gluon.utils.clip_global_norm()

    """
    def _norm(array):
        if array.stype == 'default':
            x = array.reshape((-1,))
            return ndarray.dot(x, x)
        return array.norm().square()

    assert len(arrays) > 0
    assert len(context) >= 1

    ctx = arrays[0].context
    arrays_at_all_context = arrays[0]
    # multi GPUs need to sum up each row(parameters at each context) first
    if len(context > 1):
        for i in range(len(context)):
            arrays_at_all_context += arrays[i]

    total_norm = ndarray.add_n(*[_norm(arr).as_in_context(ctx) for arr in arrays_at_all_context])
    total_norm = ndarray.sqrt(total_norm).asscalar()
    if not np.isfinite(total_norm):
        warnings.warn(UserWarning('nan or inf is detected. Clipping results will be undefined.'),
                      stacklevel=2)
    scale = max_norm * len(context) / (total_norm + 1e-8)
    if scale < 1.0:
        for arr in arrays:
            arr *= scale
    return total_norm


def train():
    """Training function."""
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr})

    train_data_loader, val_data_loader, test_data_loader \
        = dataprocessor.make_dataloader(data_train, data_val, data_test,
                                        args, num_shards=len(context))

    best_valid_bleu = 0.0
    for epoch_id in range(args.epochs):
        log_avg_loss = 0
        log_avg_gnorm = 0
        log_wc = 0
        loss_denom = 0
        step_loss = 0

        log_start_time = time.time()
        for batch_id, seqs \
                in enumerate(train_data_loader):

            src_wc, tgt_wc, bs = np.sum([(shard[2].sum(), shard[3].sum(), shard[0].shape[0])
                                         for shard in seqs], axis=0)
            src_wc = src_wc.asscalar()
            tgt_wc = tgt_wc.asscalar()

            loss_denom += tgt_wc - bs

            seqs = [[seq.as_in_context(_context) for seq in shard]
                    for _context, shard in zip(context, seqs)]

            LS = []
            with mx.autograd.record():
                for src_seq, tgt_seq, src_valid_length, tgt_valid_length in seqs:
                    out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)

                    ls = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).sum()
                    LS.append(ls * (tgt_seq.shape[1] - 1) / float(loss_denom))
            for L in LS:
                L.backward()

            grads = [p.grad(c) for p in model.collect_params().values() for c in context]
            gnorm = clip_global_norm(grads, args.clip, context=context)
            log_avg_gnorm += gnorm

            trainer.step(1.0)

            log_wc += src_wc + tgt_wc
            step_loss += sum([L.asscalar() for L in LS])
            log_avg_loss += step_loss
            loss_denom = 0
            step_loss = 0

            if (batch_id + 1) % args.log_interval == 0:
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, gnorm={:.4f} '
                             'throughput={:.2f}K wps, wc={:.2f}K'
                             .format(epoch_id, batch_id + 1, len(train_data_loader),
                                     log_avg_loss / args.log_interval,
                                     np.exp(log_avg_loss / args.log_interval),
                                     log_avg_gnorm / args.log_interval,
                                     wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0
                log_avg_gnorm = 0

        valid_loss, valid_translation_out = evaluate(val_data_loader)
        valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out)
        logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                     .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
        test_loss, test_translation_out = evaluate(test_data_loader)
        test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out)
        logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                     .format(epoch_id, test_loss, np.exp(test_loss), test_bleu_score * 100))
        dataprocessor.write_sentences(valid_translation_out,
                                      os.path.join(args.save_dir,
                                                   'epoch{:d}_valid_out.txt').format(epoch_id))
        dataprocessor.write_sentences(test_translation_out,
                                      os.path.join(args.save_dir,
                                                   'epoch{:d}_test_out.txt').format(epoch_id))
        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            save_path = os.path.join(args.save_dir, 'valid_best.params')
            logging.info('Save best parameters to {}'.format(save_path))
            model.save_parameters(save_path)
        if epoch_id + 1 >= (args.epochs * 2) // 3:
            new_lr = trainer.learning_rate * args.lr_update_factor
            logging.info('Learning rate change to {}'.format(new_lr))
            trainer.set_learning_rate(new_lr)
    if os.path.exists(os.path.join(args.save_dir, 'valid_best.params')):
        model.load_parameters(os.path.join(args.save_dir, 'valid_best.params'))
    valid_loss, valid_translation_out = evaluate(val_data_loader)
    valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out)
    logging.info('Best model valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                 .format(valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
    test_loss, test_translation_out = evaluate(test_data_loader)
    test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out)
    logging.info('Best model test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                 .format(test_loss, np.exp(test_loss), test_bleu_score * 100))
    dataprocessor.write_sentences(valid_translation_out,
                                  os.path.join(args.save_dir, 'best_valid_out.txt'))
    dataprocessor.write_sentences(test_translation_out,
                                  os.path.join(args.save_dir, 'best_test_out.txt'))


if __name__ == '__main__':
    train()
