"""Pretraining Example for BERT Model"""

import os
import time
import shutil
import logging
import argparse
import functools
import collections

import mxnet as mx
import numpy as np
from mxnet.lr_scheduler import PolyScheduler

from pretraining_utils import get_pretrain_data_npz, get_pretrain_data_text, MaskedAccuracy
from gluonnlp.utils.misc import repeat, grouper, set_seed, init_comm, logging_config, naming_convention
from gluonnlp.initializer import TruncNorm
from gluonnlp.models.bert import BertModel, BertForPretrain, get_pretrained_bert
from gluonnlp.utils.parameter import clip_grad_global_norm
try:
    import horovod.mxnet as hvd
except ImportError:
    pass

try:
    import byteps.mxnet as bps
except ImportError:
    pass

mx.npx.set_np()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt_dir',
                        help='Path to checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=250, help='Report interval')
    parser.add_argument('--ckpt_interval', type=int, default=25000, help='Checkpoint interval')
    # model
    parser.add_argument('--model_name', type=str, default='google_en_cased_bert_base',
                        help='Name of the pretrained model.')
    # training
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to pretraining corpus file. File name with wildcard such as'
                        ' dir/*.npz is accepted. Or file name with wildcard such as dir/*.txt if'
                        ' --raw is set.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size. Number of examples per gpu in a minibatch. default is 8')
    parser.add_argument('--num_accumulated', type=int, default=1,
                        help='Number of batches for gradient accumulation. '
                             'total_batch_size = batch_size_per_worker * num_worker * accumulate.')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of optimization steps')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='The optimization algorithm')
    parser.add_argument('--start_step', type=int, default=0,
                        help='Start optimization step from the checkpoint.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm.')
    parser.add_argument('--warmup_ratio', type=float, default=0.01,
                        help='Ratio of warmup steps in the learning rate scheduler.')
    parser.add_argument('--no_compute_acc', action='store_true',
                        help='skip accuracy metric computation during training')
    # debugging
    parser.add_argument('--verbose', action='store_true', help='verbose logging')
    # data pre-processing
    parser.add_argument('--num_buckets', type=int, default=1,
                        help='Number of buckets for variable length sequence sampling')
    parser.add_argument('--raw', action='store_true',
                        help='If set, both training and dev samples are generated on-the-fly '
                             'from raw texts instead of pre-processed npz files. ')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum input sequence length. Effective only if --raw is set.')
    parser.add_argument('--short_seq_prob', type=float, default=0.1,
                        help='The probability of producing sequences shorter than max_seq_length. '
                             'Effective only if --raw is set.')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15,
                        help='Probability for masks. Effective only if --raw is set.')
    parser.add_argument('--max_predictions_per_seq', type=int, default=80,
                        help='Maximum number of predictions per sequence. '
                             'Effective only if --raw is set.')
    parser.add_argument('--whole_word_mask', action='store_true',
                        help='Whether to use whole word masking rather than per-subword masking.'
                             'Effective only if --raw is set.')
    parser.add_argument('--random_next_sentence', action='store_true',
                        help='Whether to use the sentence order prediction objective as in ALBERT'
                             'Effective only if --raw is set.')
    parser.add_argument('--num_dataset_workers', type=int, default=0,
                        help='Number of workers to pre-process dataset.')
    parser.add_argument('--num_batch_workers', type=int, default=0,
                        help='Number of workers to pre-process mini-batch.')
    parser.add_argument('--circle_length', type=int, default=2,
                        help='Number of files to be read for a single GPU at the same time.')
    parser.add_argument('--repeat', type=int, default=8,
                        help='Number of times that files are repeated in each shuffle.')
    parser.add_argument('--dataset_cached', action='store_true',
                        help='Whether or not to cache the last processed training dataset.')
    parser.add_argument('--num_max_dataset_cached', type=int, default=0,
                        help='Maximum number of cached processed training dataset.')
    # communication
    parser.add_argument('--comm_backend', type=str, default='device',
                        choices=['byteps', 'horovod', 'dist_sync_device', 'device'],
                        help='Communication backend.')
    parser.add_argument('--gpus', type=str, default=None,
                        help='List of gpus to run when device or dist_sync_device is used for '
                             'communication, e.g. 0 or 0,2,5. empty means using cpu.')

    args = parser.parse_args()
    return args


def get_pretraining_model(model_name, ctx_l, max_seq_length=512):
    cfg, tokenizer, _, _ = get_pretrained_bert(
        model_name, load_backbone=False, load_mlm=False)
    cfg = BertModel.get_cfg().clone_merge(cfg)
    cfg.defrost()
    cfg.MODEL.max_length = max_seq_length
    cfg.freeze()
    model = BertForPretrain(cfg)
    model.initialize(ctx=ctx_l)
    model.hybridize()
    return cfg, tokenizer, model


def final_save(model, save_dir, tokenizer, cfg):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'model.yml'), 'w') as of:
        of.write(cfg.dump())
    tokenizer.vocab.save(os.path.join(save_dir, 'vocab.json'))
    model.backbone_model.save_parameters(os.path.join(save_dir, 'model.params'))
    logging.info('Statistics:')
    old_names = os.listdir(save_dir)
    for old_name in old_names:
        new_name, long_hash = naming_convention(save_dir, old_name)
        old_path = os.path.join(save_dir, old_name)
        new_path = os.path.join(save_dir, new_name)
        shutil.move(old_path, new_path)
        file_size = os.path.getsize(new_path)
        logging.info('\t{}/{} {} {}'.format(save_dir, new_name, long_hash, file_size))


def parameters_option(step_num, model, ckpt_dir, option='Saving'):
    """Save or load the model parameter, marked by step_num."""
    param_path = os.path.join(
        ckpt_dir, '{}.params'.format(str(step_num).zfill(7)))
    logging.info('[step {}], {} model params to/from {}.'.format(
        step_num, option, param_path))
    if option == 'Saving':
        model.save_parameters(param_path)
    elif option == 'Loading':
        model.load_parameters(param_path)
    else:
        raise NotImplementedError('Unknown Option: {}'.format(option))


def states_option(step_num, trainer, ckpt_dir, local_rank=0, option='Saving'):
    """Save or load the trainer states, marked by step_num and local rank."""
    state_path = os.path.join(ckpt_dir, '{}.states.{}'.format(
        str(step_num).zfill(7), str(local_rank).zfill(2)))
    logging.info('[step {}], {} trainer states to/from {}.'.format(
        step_num, option, state_path))
    if option == 'Saving':
        trainer.save_states(state_path)
    elif option == 'Loading':
        trainer.load_states(state_path)
    else:
        raise NotImplementedError('Unknown Option: {}'.format(option))


def train(args):
    _, num_workers, rank, local_rank, is_master_node, ctx_l = init_comm(
        args.comm_backend, args.gpus)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging_config(args.ckpt_dir,
                   name='pretrain_bert_' + str(rank),  # avoid race
                   level=level,
                   console=(local_rank == 0))
    logging.info(args)
    logging.debug('Random seed set to {}'.format(args.seed))
    set_seed(args.seed)
    logging.info('Training info: num_buckets: {}, '
                 'num_workers: {}, rank: {}'.format(
                     args.num_buckets, num_workers, rank))
    cfg, tokenizer, model = get_pretraining_model(args.model_name, ctx_l, args.max_seq_length)

    if args.raw:
        get_dataset_fn = functools.partial(get_pretrain_data_text,
                                           max_seq_length=args.max_seq_length,
                                           short_seq_prob=args.short_seq_prob,
                                           masked_lm_prob=args.masked_lm_prob,
                                           max_predictions_per_seq=args.max_predictions_per_seq,
                                           whole_word_mask=args.whole_word_mask,
                                           random_next_sentence=args.random_next_sentence,
                                           tokenizer=tokenizer,
                                           circle_length=args.circle_length,
                                           repeat=args.repeat,
                                           dataset_cached=args.dataset_cached,
                                           num_max_dataset_cached=args.num_max_dataset_cached)
    else:
        get_dataset_fn = get_pretrain_data_npz

    data_train = get_dataset_fn(args.data, args.batch_size, shuffle=True,
                                num_buckets=args.num_buckets, vocab=tokenizer.vocab,
                                num_parts=num_workers, part_idx=rank,
                                num_dataset_workers=args.num_dataset_workers,
                                num_batch_workers=args.num_batch_workers)

    param_dict = model.collect_params()
    # Do not apply weight decay to all the LayerNorm and bias
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Set grad_req if gradient accumulation is required
    params = [p for p in param_dict.values() if p.grad_req != 'null']
    num_accumulated = args.num_accumulated
    if num_accumulated > 1:
        logging.info('Using gradient accumulation. Effective global batch size = {}'
                     .format(num_accumulated * args.batch_size * len(ctx_l) * num_workers))
        for p in params:
            p.grad_req = 'add'

    num_steps = args.num_steps
    warmup_steps = int(num_steps * args.warmup_ratio)
    log_interval = args.log_interval
    save_interval = args.ckpt_interval
    logging.info('#Total Training Steps={}, Warmup Steps={}, Save Interval={}'
                 .format(num_steps, warmup_steps, save_interval))
    lr_scheduler = PolyScheduler(max_update=num_steps,
                                 base_lr=args.lr,
                                 warmup_begin_lr=0,
                                 pwr=1,
                                 final_lr=0,
                                 warmup_steps=warmup_steps,
                                 warmup_mode='linear')
    optimizer_params = {'learning_rate': args.lr,
                        'wd': args.wd,
                        'lr_scheduler': lr_scheduler,
                        }
    if args.optimizer == 'adamw':
        optimizer_params.update({'beta1': 0.9,
                                 'beta2': 0.999,
                                 'epsilon': 1e-6,
                                 'correct_bias': False,
                                 })
    if args.comm_backend == 'horovod':
        trainer = hvd.DistributedTrainer(param_dict, args.optimizer, optimizer_params)
    elif args.comm_backend == 'byteps':
        trainer = bps.DistributedTrainer(param_dict, args.optimizer, optimizer_params)
    else:
        trainer = mx.gluon.Trainer(param_dict, args.optimizer, optimizer_params,
                                   update_on_kvstore=False)
    if args.start_step:
        logging.info('Restart training from {}'.format(args.start_step))
        parameters_option(args.start_step, model, args.ckpt_dir, 'Loading')
        states_option(args.start_step, trainer, args.ckpt_dir, local_rank, 'Loading')

    if args.comm_backend == 'byteps':
        trainer._init_params()
    # backend specific implementation
    if args.comm_backend == 'horovod':
        # Horovod: fetch and broadcast parameters
        hvd.broadcast_parameters(param_dict, root_rank=0)

    # prepare the loss function
    nsp_loss_fn = mx.gluon.loss.SoftmaxCELoss()
    mlm_loss_fn = mx.gluon.loss.SoftmaxCELoss()
    nsp_loss_fn.hybridize()
    mlm_loss_fn.hybridize()

    mlm_metric = MaskedAccuracy()
    nsp_metric = MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    step_num = args.start_step
    running_mlm_loss, running_nsp_loss = 0., 0.
    running_num_tks = 0

    train_start_time = time.time()
    tic = time.time()
    # start training
    train_loop_dataloader = grouper(repeat(data_train), len(ctx_l))
    while step_num < num_steps:
        for _ in range(num_accumulated):
            sample_l = next(train_loop_dataloader)
            mlm_loss_l = []
            nsp_loss_l = []
            loss_l = []
            ns_label_list, ns_pred_list = [], []
            mask_label_list, mask_pred_list, mask_weight_list = [], [], []
            for sample, ctx in zip(sample_l, ctx_l):
                # prepare data
                (input_id, masked_id, masked_position, masked_weight, \
                    next_sentence_label, segment_id, valid_length) = sample
                input_id = input_id.as_in_ctx(ctx)
                masked_id = masked_id.as_in_ctx(ctx)
                masked_position = masked_position.as_in_ctx(ctx)
                masked_weight = masked_weight.as_in_ctx(ctx)
                next_sentence_label = next_sentence_label.as_in_ctx(ctx)
                segment_id = segment_id.as_in_ctx(ctx)
                valid_length = valid_length.as_in_ctx(ctx)

                with mx.autograd.record():
                    _, _, nsp_score, mlm_scores = model(input_id, segment_id,
                        valid_length, masked_position)
                    denominator = (masked_weight.sum() + 1e-8) * num_accumulated * len(ctx_l)
                    mlm_scores_r = mx.npx.reshape(mlm_scores, (-5, -1))
                    masked_id_r = masked_id.reshape((-1,))
                    mlm_loss = mlm_loss_fn(
                        mlm_scores_r,
                        masked_id_r,
                        masked_weight.reshape((-1, 1))).sum() / denominator
                    denominator = num_accumulated * len(ctx_l)
                    nsp_loss = nsp_loss_fn(
                        nsp_score, next_sentence_label).mean() / denominator
                    mlm_loss_l.append(mlm_loss)
                    nsp_loss_l.append(nsp_loss)
                    loss_l.append(mlm_loss + nsp_loss)
                    mask_label_list.append(masked_id_r)
                    mask_pred_list.append(mlm_scores_r)
                    mask_weight_list.append(masked_weight.reshape((-1,)))
                    ns_label_list.append(next_sentence_label)
                    ns_pred_list.append(nsp_score)

                running_num_tks += valid_length.sum().as_in_ctx(mx.cpu())

            for loss in loss_l:
                loss.backward()

            running_mlm_loss += sum([ele.as_in_ctx(mx.cpu())
                                    for ele in mlm_loss_l]).asnumpy().item()
            running_nsp_loss += sum([ele.as_in_ctx(mx.cpu())
                                    for ele in nsp_loss_l]).asnumpy().item()
            mlm_metric.update(mask_label_list, mask_pred_list, mask_weight_list)
            nsp_metric.update(ns_label_list, ns_pred_list)
        # update
        trainer.allreduce_grads()

        total_norm, ratio, is_finite = clip_grad_global_norm(
            params, args.max_grad_norm * num_workers)
        total_norm = total_norm / num_workers

        if args.comm_backend == 'horovod' or args.comm_backend == 'byteps':
            # Note that horovod.trainer._scale is default to num_workers,
            # thus trainer.update(1) will scale the gradients by 1./num_workers
            trainer.update(1, ignore_stale_grad=True)
        else:
            # gluon.trainer._scale is default to 1
            trainer.update(num_workers, ignore_stale_grad=True)

        if num_accumulated > 1:
            # set grad to zero for gradient accumulation
            model.zero_grad()

        step_num += 1
        # saving
        if step_num % save_interval == 0 or step_num >= num_steps:
            states_option(step_num, trainer, args.ckpt_dir, local_rank, 'Saving')
            if local_rank == 0:
                parameters_option(step_num, model, args.ckpt_dir, 'Saving')
        # logging
        if step_num % log_interval == 0:
            running_mlm_loss /= log_interval
            running_nsp_loss /= log_interval
            toc = time.time()
            logging.info(
                '[step {}], Loss mlm/nsp={:.5f}/{:.3f}, Acc mlm/nsp={:.3f}/{:.3f}, '
                ' LR={:.7f}, grad_norm={:.4f}. Time cost={:.2f} s,'
                ' Throughput={:.1f}K tks/s, ETA={:.2f} h'.format(
                    step_num, running_mlm_loss, running_nsp_loss,
                    mlm_metric.get()[1], nsp_metric.get()[1],
                    trainer.learning_rate, total_norm, toc - tic,
                    running_num_tks.asnumpy().item() / (toc - tic) / 1000,
                    (num_steps - step_num) / (step_num / (toc - train_start_time)) / 3600))
            mlm_metric.reset()
            nsp_metric.reset()
            tic = time.time()

            running_mlm_loss = 0
            running_nsp_loss = 0
            running_num_tks = 0

    logging.info('Finish training step: %d', step_num)

    mx.npx.waitall()
    train_end_time = time.time()
    logging.info('Train cost={:.1f} s'.format(train_end_time - train_start_time))

    if local_rank == 0:
        model_name = args.model_name.replace('google', 'gluon')
        save_dir = os.path.join(args.ckpt_dir, model_name)
        final_save(model, save_dir, tokenizer, cfg)


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    args = parse_args()
    train(args)

