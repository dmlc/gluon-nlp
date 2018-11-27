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
from mxnet import gluon
<<<<<<< HEAD
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader
from gluonnlp.data import ShardedDataLoader
import gluonnlp.data.batchify as btf
from gluonnlp.data import ConstWidthBucket, LinearWidthBucket, ExpWidthBucket, \
    FixedBucketSampler, IWSLT2015, WMT2016BPE
from gluonnlp.model import BeamSearchScorer
=======
import gluonnlp as nlp

>>>>>>> upstream/master
from gnmt import get_gnmt_encoder_decoder
from translation import NMTModel, BeamSearchTranslator
from loss import SoftmaxCEMaskedLoss
from utils import logging_config
from bleu import compute_bleu
import dataprocessor

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Google NMT model')
parser.add_argument('--dataset', type=str, default='IWSLT2015', help='Dataset to use, default is IWSLT2015.')
parser.add_argument('--src_lang', type=str, default='en', help='Source language, default is en.')
parser.add_argument('--tgt_lang', type=str, default='vi', help='Target language, default is vi.')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit, , default is 40.')
parser.add_argument('--num_hidden', type=int, default=128, help='Dimension of the rnn hidden states. default is 128.')
parser.add_argument('--num_embedding', type=int, default=128, help='Dimension of the embedding, default is 128. '
                                                                'vectors and states.')
parser.add_argument('--attention_type', type=str, default='dot', help=' Attention type, default is dot.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout), default is 0.2.')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the encoder'
                                                              ' and decoder, default is 2.')
parser.add_argument('--num_bi_layers', type=int, default=1,
                    help='number of bidirectional layers in the encoder and decoder, default is 1.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size, default is 128.')
parser.add_argument('--beam_size', type=int, default=4, help='Beam size, default is 4.')
parser.add_argument('--lp_alpha', type=float, default=1.0,
                    help='Alpha used in calculating the length penalty, default is 1.0.')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty, default is 5.')
parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size, default is 32.')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number, default is 5.')
parser.add_argument('--bucket_scheme', type=str, default='constant',
                    help='Strategy for generating bucket keys. It supports: '
                         '"constant": all the buckets have the same width; '
                         '"linear": the width of bucket increases linearly; '
                         '"exp": the width of bucket increases exponentially'
                         'default is "constant".')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing'
                                                                    'default is 0.0.')
parser.add_argument('--src_max_len', type=int, default=50, help='Maximum length of the source '
                                                                'sentence, default is 50.')
parser.add_argument('--tgt_max_len', type=int, default=50, help='Maximum length of the target '
                                                                'sentence, default is 50.')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm, default is adam.')
parser.add_argument('--lr', type=float, default=1E-3, help='Initial learning rate, default is 1E-3.')
parser.add_argument('--lr_update_factor', type=float, default=0.5,
                    help='Learning rate decay factor, default is 0.5.')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping, default is 5.0.')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval, default is 100.')
parser.add_argument('--save_dir', type=str, default='out_dir',
                    help='directory path to save the final model and training log,, default is \'out_dir\'.')
parser.add_argument('--gpus', type=str, default=None,
                    help='id of the gpus to use, eg:0,1,2,3 means use 4 GPUs. Set it to empty means to use cpu.')
args = parser.parse_args()
print(args)
logging_config(args.save_dir)


<<<<<<< HEAD
def cache_dataset(dataset, prefix):
    """Cache the processed npy dataset  the dataset into a npz

    Parameters
    ----------
    dataset : SimpleDataset
    file_path : str
    """
    if not os.path.exists(_C.CACHE_PATH):
        os.makedirs(_C.CACHE_PATH)
    src_data = np.array([ele[0] for ele in dataset])
    tgt_data = np.array([ele[1] for ele in dataset])
    np.savez(os.path.join(_C.CACHE_PATH, prefix + '.npz'), src_data=src_data, tgt_data=tgt_data)


def load_cached_dataset(prefix):
    cached_file_path = os.path.join(_C.CACHE_PATH, prefix + '.npz')
    if os.path.exists(cached_file_path):
        print('Load cached data from {}'.format(cached_file_path))
        dat = np.load(cached_file_path)
        return ArrayDataset(np.array(dat['src_data']), np.array(dat['tgt_data']))
    else:
        return None


class TrainValDataTransform(object):
    """Transform the machine translation dataset.

    Clip source and the target sentences to the maximum length. For the source sentence, append the
    EOS. For the target sentence, append BOS and EOS.

    Parameters
    ----------
    src_vocab : Vocab
    tgt_vocab : Vocab
    src_max_len : int
    tgt_max_len : int
    """
    def __init__(self, src_vocab, tgt_vocab, src_max_len, tgt_max_len):
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

    def __call__(self, src, tgt):
        if self._src_max_len > 0:
            src_sentence = self._src_vocab[src.split()[:self._src_max_len]]
        else:
            src_sentence = self._src_vocab[src.split()]
        if self._tgt_max_len > 0:
            tgt_sentence = self._tgt_vocab[tgt.split()[:self._tgt_max_len]]
        else:
            tgt_sentence = self._tgt_vocab[tgt.split()]
        src_sentence.append(self._src_vocab[self._src_vocab.eos_token])
        tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
        tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
        src_npy = np.array(src_sentence, dtype=np.int32)
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return src_npy, tgt_npy


def process_dataset(dataset, src_vocab, tgt_vocab, src_max_len=-1, tgt_max_len=-1):
    start = time.time()
    dataset_processed = dataset.transform(TrainValDataTransform(src_vocab, tgt_vocab,
                                                                src_max_len,
                                                                tgt_max_len), lazy=False)
    end = time.time()
    print('Processing Time spent: {}'.format(end - start))
    return dataset_processed


def load_translation_data(dataset, src_lang='en', tgt_lang='vi'):
    """Load translation dataset

    Parameters
    ----------
    dataset : str
    src_lang : str, default 'en'
    tgt_lang : str, default 'vi'

    Returns
    -------

    """
    common_prefix = 'IWSLT2015_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                   args.src_max_len, args.tgt_max_len)
    if dataset == 'IWSLT2015':
        data_train = IWSLT2015('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = IWSLT2015('val', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = IWSLT2015('test', src_lang=src_lang, tgt_lang=tgt_lang)

    elif dataset == 'TOY':
        common_prefix = 'TOY_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                 args.src_max_len, args.tgt_max_len)
        data_train = TOY('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = TOY('val', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = TOY('test', src_lang=src_lang, tgt_lang=tgt_lang)

    elif dataset == 'WMT2016BPE':
        print("|--------------------------------|")
        common_prefix = 'WMT2016BPE_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                 args.src_max_len, args.tgt_max_len)
        data_train = WMT2016BPE('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = WMT2016BPE('newstest2016', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = WMT2016BPE('newstest2015', src_lang=src_lang, tgt_lang=tgt_lang)

    else:
        raise NotImplementedError

    src_vocab, tgt_vocab = data_train.src_vocab, data_train.tgt_vocab
    data_train_processed = load_cached_dataset(common_prefix + '_train')
    if not data_train_processed:
        data_train_processed = process_dataset(data_train, src_vocab, tgt_vocab,
                                               args.src_max_len, args.tgt_max_len)
        cache_dataset(data_train_processed, common_prefix + '_train')
    data_val_processed = load_cached_dataset(common_prefix + '_val')
    if not data_val_processed:
        data_val_processed = process_dataset(data_val, src_vocab, tgt_vocab)
        cache_dataset(data_val_processed, common_prefix + '_val')
    data_test_processed = load_cached_dataset(common_prefix + '_test')
    if not data_test_processed:
        data_test_processed = process_dataset(data_test, src_vocab, tgt_vocab)
        cache_dataset(data_test_processed, common_prefix + '_test')
    fetch_tgt_sentence = lambda src, tgt: tgt.split()
    val_tgt_sentences = list(data_val.transform(fetch_tgt_sentence))
    test_tgt_sentences = list(data_test.transform(fetch_tgt_sentence))
    return data_train_processed, data_val_processed, data_test_processed, \
           val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab


def get_data_lengths(dataset):
    return list(dataset.transform(lambda srg, tgt: (len(srg), len(tgt))))


=======
>>>>>>> upstream/master
data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab\
    = dataprocessor.load_translation_data(dataset=args.dataset, bleu='tweaked', args=args)

dataprocessor.write_sentences(val_tgt_sentences, os.path.join(args.save_dir, 'val_gt.txt'))
dataprocessor.write_sentences(test_tgt_sentences, os.path.join(args.save_dir, 'test_gt.txt'))

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
<<<<<<< HEAD
data_val = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                          for i, ele in enumerate(data_val)])
data_test = SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                           for i, ele in enumerate(data_test)])

# set context of cpu or gpu.
if args.gpus is None:
=======
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                     for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                      for i, ele in enumerate(data_test)])
if args.gpu is None:
>>>>>>> upstream/master
    ctx = mx.cpu()
    context = [ctx]
elif args.gpus == "0":
    ctx = mx.gpu(0)
    context = [ctx]
else:
    ctx = mx.gpu(0)
    context = [mx.gpu(int(x)) for x in args.gpus.split(',')]


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


def train():
    """Training function."""
    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr})

<<<<<<< HEAD
    train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(),
                                  btf.Stack(dtype='float32'), btf.Stack(dtype='float32'))
    test_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(),
                                 btf.Stack(dtype='float32'), btf.Stack(dtype='float32'),
                                 btf.Stack())
    if args.bucket_scheme == 'constant':
        bucket_scheme = ConstWidthBucket()
    elif args.bucket_scheme == 'linear':
        bucket_scheme = LinearWidthBucket()
    elif args.bucket_scheme == 'exp':
        bucket_scheme = ExpWidthBucket(bucket_len_step=1.2)
    else:
        raise NotImplementedError
    # put train data into bucket
    train_batch_sampler = FixedBucketSampler(lengths=data_train_lengths,
                                             batch_size=args.batch_size,
                                             num_buckets=args.num_buckets,
                                             ratio=args.bucket_ratio,
                                             shuffle=True,
                                             num_shards=len(context),
                                             bucket_scheme=bucket_scheme)
    logging.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
    train_data_loader = ShardedDataLoader(data_train,
                                   batch_sampler=train_batch_sampler,
                                   batchify_fn=train_batchify_fn,
                                   num_workers=8)
    # put val data into bucket
    val_batch_sampler = FixedBucketSampler(lengths=data_val_lengths,
                                           batch_size=args.test_batch_size,
                                           num_buckets=args.num_buckets,
                                           ratio=args.bucket_ratio,
                                           shuffle=False)
    logging.info('Valid Batch Sampler:\n{}'.format(val_batch_sampler.stats()))
    val_data_loader = DataLoader(data_val,
                                 batch_sampler=val_batch_sampler,
                                 batchify_fn=test_batchify_fn,
                                 num_workers=8)
    test_batch_sampler = FixedBucketSampler(lengths=data_test_lengths,
                                            batch_size=args.test_batch_size,
                                            num_buckets=args.num_buckets,
                                            ratio=args.bucket_ratio,
                                            shuffle=False)
    logging.info('Test Batch Sampler:\n{}'.format(test_batch_sampler.stats()))
    test_data_loader = DataLoader(data_test,
                                  batch_sampler=test_batch_sampler,
                                  batchify_fn=test_batchify_fn,
                                  num_workers=8)
=======
    train_data_loader, val_data_loader, test_data_loader \
        = dataprocessor.make_dataloader(data_train, data_val, data_test, args)

>>>>>>> upstream/master
    best_valid_bleu = 0.0
    for epoch_id in range(args.epochs):

        log_avg_loss = 0
        log_avg_gnorm = 0
        log_wc = 0
        log_start_time = time.time()

        for batch_id, seqs \
                in enumerate(train_data_loader):

            src_wc, tgt_wc, bs = np.sum([(shard[2].sum(), shard[3].sum(), shard[0].shape[0])
                                         for shard in seqs], axis=0)

            seqs = [[seq.as_in_context(_context) for seq in shard]
                    for _context, shard in zip(context, seqs)]
            src_wc = src_wc.asscalar()
            tgt_wc = tgt_wc.asscalar()
            LS = []

            with mx.autograd.record():
                for src_seq, tgt_seq, src_valid_length, tgt_valid_length in seqs:
                    out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
                    loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean()
                    loss = loss * (tgt_seq.shape[1] - 1) / (tgt_valid_length - 1).mean()
                    LS.append(loss)
            print("|---- context is :" + str(context))
            print("|---- LS is :"+str(LS))
            for L in LS:
                L.backward()

            grads = [p.grad(ctx) for p in model.collect_params().values()]
            gnorm = gluon.utils.clip_global_norm(grads, args.clip)
            trainer.step(1)

            # calculate the loss at all contexts
            step_loss = sum([L.asscalar() for L in LS])
            log_avg_loss += step_loss / len(context)

            log_avg_gnorm += gnorm
            log_wc += src_wc + tgt_wc

            if (batch_id + 1) % args.log_interval == 0:
                wps = log_wc / (time.time() - log_start_time)

                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, gnorm={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'
                             .format(epoch_id, batch_id + 1, len(train_data_loader),
                                     log_avg_loss / args.log_interval,
                                     np.exp(log_avg_loss / args.log_interval),
                                     log_avg_gnorm / args.log_interval,
                                     wps / 1000, log_wc / 1000))

                log_start_time = time.time()
                log_avg_loss = 0
                log_avg_gnorm = 0
                log_wc = 0

        valid_loss, valid_translation_out = evaluate(val_data_loader)
        valid_bleu_score, _, _, _, _ = compute_bleu([val_tgt_sentences], valid_translation_out)
        logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                     .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))

        test_loss, test_translation_out = evaluate(test_data_loader)
        test_bleu_score, _, _, _, _ = compute_bleu([test_tgt_sentences], test_translation_out)
        logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                     .format(epoch_id, test_loss, np.exp(test_loss), test_bleu_score * 100))
<<<<<<< HEAD

        write_sentences(valid_translation_out,
                        os.path.join(args.save_dir, 'epoch{:d}_valid_out.txt').format(epoch_id))
        write_sentences(test_translation_out,
                        os.path.join(args.save_dir, 'epoch{:d}_test_out.txt').format(epoch_id))

=======
        dataprocessor.write_sentences(valid_translation_out,
                                      os.path.join(args.save_dir,
                                                   'epoch{:d}_valid_out.txt').format(epoch_id))
        dataprocessor.write_sentences(test_translation_out,
                                      os.path.join(args.save_dir,
                                                   'epoch{:d}_test_out.txt').format(epoch_id))
>>>>>>> upstream/master
        if valid_bleu_score > best_valid_bleu:
            best_valid_bleu = valid_bleu_score
            save_path = os.path.join(args.save_dir, 'valid_best.params')
            logging.info('Save best parameters to {}'.format(save_path))
            model.save_parameters(save_path)

        if epoch_id + 1 >= (args.epochs * 2) // 3:
            new_lr = trainer.learning_rate * args.lr_update_factor
            logging.info('Learning rate change to {}'.format(new_lr))
            trainer.set_learning_rate(new_lr)

    # to show the best result at the end
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