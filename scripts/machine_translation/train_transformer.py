"""
Transformer
=================================

This example shows how to implement the Transformer model with GluonNLP Toolkit.

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones,
          Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6000--6010},
  year={2017}
}
"""

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
import itertools
import math
import numpy as np
import mxnet as mx
from mxnet import gluon
from gluonnlp.models.transformer import TransformerModel
from gluonnlp.utils.misc import logging_config, AverageSGDTracker, count_parameters,\
    md5sum, grouper, init_comm
from gluonnlp.data.sampler import (
    ConstWidthBucket,
    LinearWidthBucket,
    ExpWidthBucket,
    FixedBucketSampler,
    BoundedBudgetSampler,
    ShardedIterator
)
import gluonnlp.data.batchify as bf
from gluonnlp.data import Vocab
from gluonnlp.data import tokenizers
from gluonnlp.data.tokenizers import BaseTokenizerWithVocab, huggingface
from gluonnlp.lr_scheduler import InverseSquareRootScheduler
from gluonnlp.loss import LabelSmoothCrossEntropyLoss
try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

mx.npx.set_np()


CACHE_PATH = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', 'cached'))
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Transformer for Neural Machine Translation.')
    parser.add_argument('--train_src_corpus', type=str,
                        help='The source training corpus.')
    parser.add_argument('--train_tgt_corpus', type=str,
                        help='The target training corpus.')
    parser.add_argument('--dev_src_corpus', type=str,
                        help='The source dev corpus.')
    parser.add_argument('--dev_tgt_corpus', type=str,
                        help='The target dev corpus.')
    parser.add_argument('--src_tokenizer', choices=['spm',
                                                    'subword_nmt',
                                                    'yttm',
                                                    'hf_bytebpe',
                                                    'hf_wordpiece',
                                                    'hf_bpe',
                                                    'whitespace'],
                        default='whitespace', type=str,
                        help='The source tokenizer. '
                             'Whitespace tokenizer supports processing pre-encoded corpus, '
                             'and the tokenizers besides whitespace supports online encoding.')
    parser.add_argument('--tgt_tokenizer', choices=['spm',
                                                    'subword_nmt',
                                                    'yttm',
                                                    'hf_bytebpe',
                                                    'hf_wordpiece',
                                                    'hf_bpe',
                                                    'whitespace'],
                        default='whitespace', type=str,
                        help='The target tokenizer.')
    parser.add_argument('--src_subword_model_path', type=str,
                        help='Path to the source subword model.')
    parser.add_argument('--src_vocab_path', type=str,
                        help='Path to the source vocab.')
    parser.add_argument('--tgt_subword_model_path', type=str,
                        help='Path to the target subword model.')
    parser.add_argument('--tgt_vocab_path', type=str,
                        help='Path to the target vocab.')
    parser.add_argument('--seed', type=int, default=100, help='The random seed.')
    parser.add_argument('--epochs', type=int, default=30, help='Upper epoch limit, '
                        'the model will keep training when epochs < 0 and max_update < 0.')
    parser.add_argument('--max_update', type=int, default=-1,
                        help='Max update steps, when max_update > 0, epochs will be set to -1, '
                             'each update step contains gpu_num * num_accumulated batches.')
    parser.add_argument('--save_interval_update', type=int, default=500,
                         help='Update interval of saving checkpoints while using max_update.')
    parser.add_argument('--cfg', type=str, default='transformer_base',
                        help='Configuration of the transformer model. '
                             'You may select a yml file or use the prebuild configurations.')
    parser.add_argument('--label_smooth_alpha', type=float, default=0.1,
                        help='Weight of label smoothing')
    parser.add_argument('--sampler', type=str, choices=['BoundedBudgetSampler',
                                                        'FixedBucketSampler'],
                        default='FixedBucketSampler', help='Type of sampler')
    parser.add_argument('--batch_size', type=int, default=2700,
                        help='Batch size. Number of tokens per gpu in a minibatch.')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='Batch size for evaluation.')
    parser.add_argument('--num_buckets', type=int, default=20, help='Bucket number.')
    parser.add_argument('--bucket_scheme', type=str, default='exp',
                        help='Strategy for generating bucket keys. It supports: '
                             '"constant": all the buckets have the same width; '
                             '"linear": the width of bucket increases linearly; '
                             '"exp": the width of bucket increases exponentially')
    parser.add_argument('--bucket_ratio', type=float, default=0.0,
                        help='Ratio for increasing the throughput of the bucketing')
    parser.add_argument('--max_num_tokens', type=int, default=-1,
                        help='max tokens num of each batch, applicable while using BoundedBudgetSampler')
    parser.add_argument('--max_num_sentences', type=int, default=-1,
                        help='max sentences num of each batch, applicable while using BoundedBudgetSampler')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='The learning rate at the end of the warmup stage. '
                             'If it is not given, we will use the formula suggested in the '
                             'original Transformer paper:'
                             ' 1.0 / sqrt(d_model) / sqrt(warmup_steps). '
                             'Otherwise, we will use the given lr as the final learning rate in '
                             'the warmup phase.')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='number of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument('--warmup_init_lr', type=float, default=0.0,
                        help='Initial learning rate at the beginning of the warm-up stage')
    parser.add_argument('--num_accumulated', type=int, default=32,
                        help='Number of steps to accumulate the gradients. '
                             'This is useful to mimic large batch training with limited gpu memory')
    parser.add_argument('--magnitude', type=float, default=3.0,
                        help='Magnitude of Xavier initialization')
    parser.add_argument('--num_averages', type=int, default=-1,
                        help='Perform final testing based on the '
                             'average of last num_averages checkpoints. '
                             'Use num_average will cause extra gpu memory usage.')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--save_dir', type=str, default='transformer_out',
                        help='directory path to save the final model and training log')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use dtype float16')
    parser.add_argument('--comm_backend', type=str, default='device',
                        choices=['horovod', 'dist_sync_device', 'device'],
                        help='Communication backend.')
    parser.add_argument('--gpus', type=str,
                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
    args = parser.parse_args()
    if args.max_update > 0:
        args.epochs = -1
    logging_config(args.save_dir, console=True)
    logging.info(args)
    return args


def validation(model, data_loader, ctx_l):
    """Validate the model on the dataset

    Parameters
    ----------
    model : TransformerModel
        The transformer model
    data_loader : DataLoader
        DataLoader
    ctx_l : list
        List of mx.ctx.Context


    Returns
    -------
    avg_nll_loss : float
        The average negative log-likelihood loss
    """
    avg_nll_loss = mx.np.array(0, dtype=np.float32, ctx=mx.cpu())
    ntokens = 0
    for sample_data_l in grouper(data_loader, len(ctx_l)):
        loss_l = []
        ntokens += sum([ele[3].sum().asnumpy() - ele[0].shape[0] for ele in sample_data_l
                        if ele is not None])
        for sample_data, ctx in zip(sample_data_l, ctx_l):
            if sample_data is None:
                continue
            src_token_ids, tgt_token_ids, src_valid_length, tgt_valid_length, sample_ids = sample_data
            src_token_ids = src_token_ids.as_in_ctx(ctx)
            tgt_token_ids = tgt_token_ids.as_in_ctx(ctx)
            src_valid_length = src_valid_length.as_in_ctx(ctx)
            tgt_valid_length = tgt_valid_length.as_in_ctx(ctx)
            tgt_pred = model(src_token_ids, src_valid_length, tgt_token_ids[:, :-1],
                             tgt_valid_length - 1)
            tgt_labels = tgt_token_ids[:, 1:]
            tgt_pred_logits = mx.npx.log_softmax(tgt_pred, axis=-1)
            nll_loss = - mx.npx.pick(tgt_pred_logits, tgt_labels, axis=-1)
            loss = mx.npx.sequence_mask(nll_loss,
                                        sequence_length=tgt_valid_length - 1,
                                        use_sequence_length=True,
                                        axis=1)
            loss_l.append(loss.sum())
        avg_nll_loss += sum([loss.as_in_ctx(mx.cpu()) for loss in loss_l])
        mx.npx.waitall()
    avg_loss = avg_nll_loss.asnumpy() / ntokens
    return avg_loss


def load_dataset_with_cache(src_corpus_path: str,
                            tgt_corpus_path: str,
                            src_tokenizer: BaseTokenizerWithVocab,
                            tgt_tokenizer: BaseTokenizerWithVocab,
                            overwrite_cache: bool,
                            local_rank: int):
    # TODO online h5py multi processing encode (Tao)
    src_md5sum = md5sum(src_corpus_path)
    tgt_md5sum = md5sum(tgt_corpus_path)
    cache_filepath = os.path.join(CACHE_PATH,
                                  '{}_{}.cache.npz'.format(src_md5sum[:6], tgt_md5sum[:6]))
    if os.path.exists(cache_filepath) and not overwrite_cache:
        if local_rank == 0:
            logging.info('Load cache from {}'.format(cache_filepath))
        npz_data = np.load(cache_filepath, allow_pickle=True)
        src_data, tgt_data = npz_data['src_data'][:], npz_data['tgt_data'][:]
    else:
        assert src_tokenizer.vocab.eos_id is not None,\
            'You will need to add the EOS token to the vocabulary used in the tokenizer of ' \
            'the source language.'
        assert tgt_tokenizer.vocab.bos_id is not None and tgt_tokenizer.vocab.eos_id is not None, \
            'You will need to add both the BOS token and the EOS tokens to the vocabulary used ' \
            'in the tokenizer of the target language.'
        src_data = []
        tgt_data = []
        # TODO(sxjscience) Optimize the speed of converting to cache
        with open(src_corpus_path) as f:
            for line in f:
                sample = np.array(src_tokenizer.encode(line.strip(), output_type=int) +
                                  [src_tokenizer.vocab.eos_id], dtype=np.int32)
                src_data.append(sample)
        with open(tgt_corpus_path) as f:
            for line in f:
                sample = np.array([tgt_tokenizer.vocab.bos_id] +
                                  tgt_tokenizer.encode(line.strip(), output_type=int) +
                                  [tgt_tokenizer.vocab.eos_id], dtype=np.int32)
                tgt_data.append(sample)
        src_data = np.array(src_data)
        tgt_data = np.array(tgt_data)
        np.savez(cache_filepath, src_data=src_data, tgt_data=tgt_data)
    return src_data, tgt_data


def create_tokenizer(tokenizer_type, model_path, vocab_path):
    if tokenizer_type == 'whitespace':
        return tokenizers.create(tokenizer_type, vocab=Vocab.load(vocab_path))
    elif tokenizer_type == 'spm':
        return tokenizers.create(tokenizer_type, model_path=model_path, vocab=vocab_path)
    elif tokenizer_type == 'subword_nmt':
        return tokenizers.create(tokenizer_type, model_path=model_path, vocab=vocab_path)
    elif tokenizer_type == 'yttm':
        return tokenizers.create(tokenizer_type, model_path=model_path)
    elif tokenizer_type in ['hf_bytebpe', 'hf_wordpiece', 'hf_bpe']:
        if huggingface.is_new_version_model_file(model_path):
            return tokenizers.create('hf_tokenizer', model_path=model_path, vocab=vocab_path)
        elif tokenizer_type == 'hf_bytebpe':
            return tokenizers.create(tokenizer_type, merges_file=model_path, vocab_file=vocab_path)
        elif tokenizer_type == 'hf_wordpiece':
            return tokenizers.create(tokenizer_type, vocab_file=vocab_path)
        elif tokenizer_type == 'hf_bpe':
            return tokenizers.create(tokenizer_type, merges_file=model_path, vocab_file=vocab_path)
    else:
        raise NotImplementedError


def train(args):
    _, num_parts, rank, local_rank, _, ctx_l = init_comm(
        args.comm_backend, args.gpus)
    src_tokenizer = create_tokenizer(args.src_tokenizer,
                                     args.src_subword_model_path,
                                     args.src_vocab_path)
    tgt_tokenizer = create_tokenizer(args.tgt_tokenizer,
                                     args.tgt_subword_model_path,
                                     args.tgt_vocab_path)
    src_vocab = src_tokenizer.vocab
    tgt_vocab = tgt_tokenizer.vocab
    train_src_data, train_tgt_data = load_dataset_with_cache(args.train_src_corpus,
                                                             args.train_tgt_corpus,
                                                             src_tokenizer,
                                                             tgt_tokenizer,
                                                             args.overwrite_cache,
                                                             local_rank)
    dev_src_data, dev_tgt_data = load_dataset_with_cache(args.dev_src_corpus,
                                                         args.dev_tgt_corpus,
                                                         src_tokenizer,
                                                         tgt_tokenizer,
                                                         args.overwrite_cache,
                                                         local_rank)
    data_train = gluon.data.SimpleDataset(
        [(src_tokens, tgt_tokens, len(src_tokens), len(tgt_tokens), i)
         for i, (src_tokens, tgt_tokens) in enumerate(zip(train_src_data, train_tgt_data))])
    data_val = gluon.data.SimpleDataset(
        [(src_tokens, tgt_tokens, len(src_tokens), len(tgt_tokens), i)
         for i, (src_tokens, tgt_tokens) in enumerate(zip(dev_src_data, dev_tgt_data))])
    # Construct the model + loss function
    if args.cfg.endswith('.yml'):
        cfg = TransformerModel.get_cfg().clone_merge(args.cfg)
    else:
        cfg = TransformerModel.get_cfg(args.cfg)
    cfg.defrost()
    cfg.MODEL.src_vocab_size = len(src_vocab)
    cfg.MODEL.tgt_vocab_size = len(tgt_vocab)
    if args.fp16:
        raise NotImplementedError
#        cfg.MODEL.dtype = 'float16'
    cfg.freeze()
    model = TransformerModel.from_cfg(cfg)
    model.initialize(mx.init.Xavier(magnitude=args.magnitude),
                     ctx=ctx_l)
    model.hybridize()
    if local_rank == 0:
        logging.info(model)
    with open(os.path.join(args.save_dir, 'config.yml'), 'w') as cfg_f:
        cfg_f.write(cfg.dump())
    label_smooth_loss = LabelSmoothCrossEntropyLoss(num_labels=len(tgt_vocab),
                                                    alpha=args.label_smooth_alpha,
                                                    from_logits=False)
    label_smooth_loss.hybridize()
    rescale_loss = 100.0
    
    if args.comm_backend == 'horovod':
        hvd.broadcast_parameters(model.collect_params(), root_rank=0)
    
    # Construct the trainer
    # TODO(sxjscience) Support AMP
    if args.lr is None:
        base_lr = 2.0 / math.sqrt(args.num_units) / math.sqrt(args.warmup_steps)
    else:
        base_lr = args.lr
    lr_scheduler = InverseSquareRootScheduler(warmup_steps=args.warmup_steps, base_lr=base_lr,
                                              warmup_init_lr=args.warmup_init_lr)
    trainer_settings = (model.collect_params(), 'adam',
                        {'learning_rate': args.lr, 'beta1': 0.9,
                         'beta2': 0.98, 'epsilon': 1e-9, 'lr_scheduler': lr_scheduler})
    if args.comm_backend == 'horovod':
        trainer = hvd.DistributedTrainer(*trainer_settings)
    else:
        trainer = gluon.Trainer(*trainer_settings)
    # Load Data
    if args.sampler == 'BoundedBudgetSampler':
        train_batch_sampler = BoundedBudgetSampler(lengths=[(ele[2], ele[3]) for ele in data_train],
                                                   max_num_tokens=args.max_num_tokens,
                                                   max_num_sentences=args.max_num_sentences,
                                                   seed=args.seed)
        if num_parts > 1:
            train_batch_sampler = ShardedIterator(train_batch_sampler, num_parts=num_parts,
                                                  part_index=rank)
    elif args.sampler == 'FixedBucketSampler':
        if args.comm_backend == 'horovod':
            raise NotImplementedError('FixedBucketSampler does not support horovod at present')

        if args.bucket_scheme == 'constant':
            bucket_scheme = ConstWidthBucket()
        elif args.bucket_scheme == 'linear':
            bucket_scheme = LinearWidthBucket()
        elif args.bucket_scheme == 'exp':
            bucket_scheme = ExpWidthBucket(bucket_len_step=1.2)
        else:
            raise NotImplementedError
        # TODO(sxjscience) Support auto-bucket-size tuning
        train_batch_sampler = FixedBucketSampler(lengths=[(ele[2], ele[3]) for ele in data_train],
                                                 batch_size=args.batch_size,
                                                 num_buckets=args.num_buckets,
                                                 ratio=args.bucket_ratio,
                                                 shuffle=True,
                                                 use_average_length=True,
                                                 bucket_scheme=bucket_scheme,
                                                 seed=args.seed)
    else:
        raise NotImplementedError

    logging.info(train_batch_sampler)

    batchify_fn = bf.Tuple(bf.Pad(), bf.Pad(), bf.Stack(), bf.Stack(), bf.Stack())
    train_data_loader = gluon.data.DataLoader(data_train,
                                              batch_sampler=train_batch_sampler,
                                              batchify_fn=batchify_fn,
                                              num_workers=0)

    val_data_loader = gluon.data.DataLoader(data_val,
                                            batch_size=args.val_batch_size,
                                            batchify_fn=batchify_fn,
                                            num_workers=0,
                                            shuffle=False)
    for v in model.collect_params().values():
        if v.grad_req != 'null':
            v.grad_req = 'add'
    model.zero_grad()
    model_averager = AverageSGDTracker(model.collect_params())
    log_start_time = time.time()
    num_params, num_fixed_params = None, None
    # TODO(sxjscience) Add a log metric class
    accum_count = 0
    loss_denom = 0
    n_train_iters = 0
    log_wc = 0
    log_avg_loss = 0.0
    log_loss_denom = 0
    epoch_id = 0
    while (args.epochs < 0 or epoch_id < args.epochs): # when args.epochs < 0, the model will keep training
        n_epoch_train_iters = 0
        processed_batch_num = 0
        train_multi_data_loader = grouper(train_data_loader, len(ctx_l))
        is_last_batch = False
        sample_data_l = next(train_multi_data_loader)
        while not is_last_batch:
            processed_batch_num += len(sample_data_l)
            loss_l = []
            for sample_data, ctx in zip(sample_data_l, ctx_l):
                if sample_data is None:
                    continue
                src_token_ids, tgt_token_ids, src_valid_length, tgt_valid_length, sample_ids = sample_data
                src_wc, tgt_wc, bs = src_valid_length.sum(), tgt_valid_length.sum(), src_token_ids.shape[0]
                loss_denom += tgt_wc - bs
                log_loss_denom += tgt_wc - bs
                log_wc += src_wc + tgt_wc
                src_token_ids = src_token_ids.as_in_ctx(ctx)
                tgt_token_ids = tgt_token_ids.as_in_ctx(ctx)
                src_valid_length = src_valid_length.as_in_ctx(ctx)
                tgt_valid_length = tgt_valid_length.as_in_ctx(ctx)
                with mx.autograd.record():
                    tgt_pred = model(src_token_ids, src_valid_length, tgt_token_ids[:, :-1],
                                     tgt_valid_length - 1)
                    tgt_labels = tgt_token_ids[:, 1:]
                    loss = label_smooth_loss(tgt_pred, tgt_labels)
                    loss = mx.npx.sequence_mask(loss,
                                                sequence_length=tgt_valid_length - 1,
                                                use_sequence_length=True,
                                                axis=1)
                    loss_l.append(loss.sum() / rescale_loss)
            for l in loss_l:
                l.backward()
            accum_count += 1
            try:
                sample_data_l = next(train_multi_data_loader)
            except StopIteration:
                is_last_batch = True
            if local_rank == 0 and num_params is None:
                num_params, num_fixed_params = count_parameters(model.collect_params())
                logging.info('Total Number of Parameters (not-fixed/fixed): {}/{}'
                             .format(num_params, num_fixed_params))
            sum_loss = sum([l.as_in_ctx(mx.cpu()) for l in loss_l]) * rescale_loss
            log_avg_loss += sum_loss
            mx.npx.waitall()
            if accum_count == args.num_accumulated or is_last_batch:
                # Update the parameters
                n_train_iters += 1
                n_epoch_train_iters += 1
                trainer.step(loss_denom.asnumpy() / rescale_loss)
                accum_count = 0
                loss_denom = 0
                model.zero_grad()
                if (args.epochs > 0 and epoch_id >= args.epochs - args.num_averages) or \
                   (args.max_update > 0 and n_train_iters >= args.max_update - args.num_averages * args.save_interval_update):
                    model_averager.step()
                if local_rank == 0 and \
                   (n_epoch_train_iters % args.log_interval == 0 or is_last_batch):
                    log_end_time = time.time()
                    log_wc = log_wc.asnumpy()
                    wps = log_wc / (log_end_time - log_start_time)
                    log_avg_loss = (log_avg_loss / log_loss_denom).asnumpy()
                    logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, '
                                 'throughput={:.2f}K wps, wc={:.2f}K, LR={}'
                                 .format(epoch_id, processed_batch_num * num_parts,
                                         len(train_data_loader), log_avg_loss, np.exp(log_avg_loss),
                                         wps / 1000, log_wc / 1000, trainer.learning_rate))
                    log_start_time = time.time()
                    log_avg_loss = 0
                    log_loss_denom = 0
                    log_wc = 0
                if local_rank == 0 and \
                   (args.max_update > 0 and n_train_iters % args.save_interval_update == 0):
                    n_update = n_train_iters // args.save_interval_update
                    model.save_parameters(os.path.join(args.save_dir,
                                                       'update{:d}.params'.format(n_update)),
                                          deduplicate=True)
                    avg_valid_loss = validation(model, val_data_loader, ctx_l)
                    logging.info('[Update {}] validation loss/ppl={:.4f}/{:.4f}'
                                 .format(n_update, avg_valid_loss, np.exp(avg_valid_loss)))
                if args.max_update > 0 and n_train_iters >= args.max_update:
                    break
        if local_rank == 0:
            model.save_parameters(os.path.join(args.save_dir,
                                               'epoch{:d}.params'.format(epoch_id)),
                                  deduplicate=True)
            avg_valid_loss = validation(model, val_data_loader, ctx_l)
            logging.info('[Epoch {}] validation loss/ppl={:.4f}/{:.4f}'
                         .format(epoch_id, avg_valid_loss, np.exp(avg_valid_loss)))

        if args.max_update > 0 and n_train_iters >= args.max_update:
            break
        epoch_id += 1

    if args.num_averages > 0:
        model_averager.copy_back(model.collect_params())  # TODO(sxjscience) Rewrite using update
        model.save_parameters(os.path.join(args.save_dir, 'average.params'),
                              deduplicate=True)


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    random.seed(args.seed)
    train(args)
