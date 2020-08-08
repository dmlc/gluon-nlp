"""Pretraining Example for Electra Model on the OpenWebText dataset"""

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

from sklearn import metrics
from pretraining_utils import ElectraMasker, get_pretrain_data_npz, get_pretrain_data_text
from gluonnlp.utils.misc import repeat, grouper, set_seed, init_comm, logging_config, naming_convention
from gluonnlp.initializer import TruncNorm
from gluonnlp.models.electra import ElectraModel, ElectraForPretrain, get_pretrained_electra
from gluonnlp.utils.parameter import clip_grad_global_norm
try:
    import horovod.mxnet as hvd
except ImportError:
    pass

mx.npx.set_np()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name', type=str, default='google_electra_small',
                        help='Name of the pretrained model.')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to pretraining corpus file. File name with wildcard such as'
                        ' dir/*.npz is accepted. Or file name with wildcard such as dir/*.txt if'
                        ' --from_raw_text is set.')
    parser.add_argument('--output_dir', type=str, default='electra_owt',
                        help='The output directory where the model params will be written.'
                             ' default is squad_out')
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--log_interval', type=int,
                        default=100, help='The logging interval.')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='the number of steps to save model parameters.'
                        'default is every epoch')
    # Data Loading from npz, need to be same as pretraining example
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='The maximum total input sequence length after tokenization.'
                             'Sequences longer than this will be truncated, and sequences shorter '
                             'than this will be padded. default is 128')
    parser.add_argument("--do_lower_case", dest='do_lower_case',
                        action="store_true", help="Lower case input text. Default is True")
    parser.add_argument("--no_lower_case", dest='do_lower_case',
                        action='store_false', help="Don't lower case input text.")
    parser.add_argument('--mask_prob', type=float, default=0.15,
                        help='mask probability for generator input')
    parser.set_defaults(do_lower_case=True)
    parser.add_argument('--num_dataset_workers', type=int, default=4,
                        help='Number of workers to pre-process dataset.')
    parser.add_argument('--num_batch_workers', type=int, default=2,
                        help='Number of workers to pre-process mini-batch.')
    parser.add_argument('--num_buckets', type=int, default=1,
                        help='Number of buckets for variable length sequence sampling')
    # Data pre-processing from raw text. the below flags are only valid if --from_raw_text is set
    parser.add_argument('--from_raw_text', action='store_true',
                        help='If set, both training and dev samples are generated on-the-fly '
                             'from raw texts instead of pre-processed npz files. ')
    parser.add_argument("--short_seq_prob", type=float, default=0.05,
                        help='The probability of sampling sequences '
                             'shorter than the max_seq_length.')
    parser.add_argument("--cached_file_path", default=None,
                        help='Directory for saving preprocessed features')
    parser.add_argument('--circle_length', type=int, default=2,
                        help='Number of files to be read for a single GPU at the same time.')
    parser.add_argument('--repeat', type=int, default=8,
                        help='Number of times that files are repeated in each shuffle.')
    # Optimization
    parser.add_argument('--num_train_steps', type=int, default=1000000,
                        help='The number of training steps. Note that epochs will be ignored '
                             'if training steps are set')
    parser.add_argument('--warmup_steps', type=int, default=10000,
                        help='warmup steps. Note that either warmup_steps or warmup_ratio is set.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Ratio of warmup steps in the learning rate scheduler.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size. Number of examples per gpu in a minibatch. default is 8')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm.')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimization algorithm. default is adamw')
    parser.add_argument('--lr_decay_power', type=float, default=1.0,
                        help="Decay power for layer-wise learning rate")
    parser.add_argument('--num_accumulated', type=int, default=1,
                        help='The number of batches for gradients accumulation to '
                             'simulate large batch size.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate. default is 5e-4')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--start_step', type=int, default=0,
                        help='Start optimization step from the checkpoint.')
    # Modle Configuration
    parser.add_argument('--disc_weight', type=float, default=50.0,
                        help='loss wight for discriminator')
    parser.add_argument('--gen_weight', type=float, default=1.0,
                        help='loss wight for generator')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                        help='dropout of hidden layer')
    parser.add_argument('--attention_dropout_prob', type=float, default=0.1,
                        help='dropout of attention layer')
    parser.add_argument('--generator_units_scale', type=float, default=None,
                        help='The scale size of the generator units')
    parser.add_argument('--generator_layers_scale', type=float, default=None,
                        help='The scale size of the generator layer')
    # Communication
    parser.add_argument('--comm_backend', type=str, default='device',
                        choices=['horovod', 'dist_sync_device', 'device'],
                        help='Communication backend.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    args = parser.parse_args()
    return args


def get_pretraining_model(model_name, ctx_l,
                          max_seq_length=128,
                          hidden_dropout_prob=0.1,
                          attention_dropout_prob=0.1,
                          generator_units_scale=None,
                          generator_layers_scale=None):
    """
    A Electra Pretrain Model is built with a generator and a discriminator, in which
    the generator has the same embedding as the discriminator but different backbone.
    """
    cfg, tokenizer, _, _ = get_pretrained_electra(
        model_name, load_backbone=False)
    cfg = ElectraModel.get_cfg().clone_merge(cfg)
    cfg.defrost()
    cfg.MODEL.hidden_dropout_prob = hidden_dropout_prob
    cfg.MODEL.attention_dropout_prob = attention_dropout_prob
    cfg.MODEL.max_length = max_seq_length
    # Keep the original generator size if not designated
    if generator_layers_scale:
        cfg.MODEL.generator_layers_scale = generator_layers_scale
    if generator_units_scale:
        cfg.MODEL.generator_units_scale = generator_units_scale
    cfg.freeze()

    model = ElectraForPretrain(cfg,
                               uniform_generator=False,
                               tied_generator=False,
                               tied_embeddings=True,
                               disallow_correct=False,
                               weight_initializer=TruncNorm(stdev=0.02))
    model.initialize(ctx=ctx_l)
    model.hybridize()
    return cfg, tokenizer, model


ElectraOutput = collections.namedtuple('ElectraOutput',
                                       ['mlm_scores',
                                        'rtd_scores',
                                        'rtd_labels',
                                        'corrupted_tokens'])


def final_save(model, save_dir, tokenizer):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'model.yml'), 'w') as of:
        of.write(model.disc_cfg.dump())
    tokenizer.vocab.save(os.path.join(save_dir, 'vocab.json'))
    model.disc_backbone.save_parameters(os.path.join(save_dir, 'model.params'))
    model.discriminator.save_parameters(os.path.join(save_dir, 'disc_model.params'))
    model.generator.save_parameters(os.path.join(save_dir, 'gen_model.params'))

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
        return param_path
    elif option == 'Loading':
        model.load_parameters(param_path)
        return model
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
        return state_path
    elif option == 'Loading':
        trainer.load_states(state_path)
        return trainer
    else:
        raise NotImplementedError('Unknown Option: {}'.format(option))


def train(args):
    store, num_workers, rank, local_rank, is_master_node, ctx_l = init_comm(
        args.comm_backend, args.gpus)
    logging.info('Training info: num_buckets: {}, '
                 'num_workers: {}, rank: {}'.format(
                     args.num_buckets, num_workers, rank))
    cfg, tokenizer, model = get_pretraining_model(args.model_name, ctx_l,
                                                  args.max_seq_length,
                                                  args.hidden_dropout_prob,
                                                  args.attention_dropout_prob,
                                                  args.generator_units_scale,
                                                  args.generator_layers_scale)
    data_masker = ElectraMasker(
        tokenizer, args.max_seq_length, args.mask_prob)
    if args.from_raw_text:
        if args.cached_file_path and not os.path.exists(args.cached_file_path):
            os.mkdir(args.cached_file_path)
        get_dataset_fn = functools.partial(get_pretrain_data_text,
                                           max_seq_length=args.max_seq_length,
                                           short_seq_prob=args.short_seq_prob,
                                           tokenizer=tokenizer,
                                           circle_length=args.circle_length,
                                           repeat=args.repeat,
                                           cached_file_path=args.cached_file_path)

        logging.info('Processing and loading the training dataset from raw text.')

    else:
        logging.info('Loading the training dataset from local Numpy file.')
        get_dataset_fn = get_pretrain_data_npz

    data_train = get_dataset_fn(args.data, args.batch_size, shuffle=True,
                                num_buckets=args.num_buckets, vocab=tokenizer.vocab,
                                num_parts=num_workers, part_idx=rank,
                                num_dataset_workers=args.num_dataset_workers,
                                num_batch_workers=args.num_batch_workers)

    logging.info('Creating distributed trainer...')
    param_dict = model.collect_params()
    # Do not apply weight decay to all the LayerNorm and bias
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in param_dict.values() if p.grad_req != 'null']
    # Set grad_req if gradient accumulation is required
    if args.num_accumulated > 1:
        logging.info('Using gradient accumulation. Effective global batch size = {}'
                     .format(args.num_accumulated * args.batch_size * len(ctx_l) * num_workers))
        for p in params:
            p.grad_req = 'add'
    # backend specific implementation
    if args.comm_backend == 'horovod':
        # Horovod: fetch and broadcast parameters
        hvd.broadcast_parameters(param_dict, root_rank=0)

    num_train_steps = args.num_train_steps
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(num_train_steps * args.warmup_ratio)
    assert warmup_steps is not None, 'Must specify either warmup_steps or warmup_ratio'
    log_interval = args.log_interval
    save_interval = args.save_interval if args.save_interval is not None\
        else num_train_steps // 50
    logging.info('#Total Training Steps={}, Warmup={}, Save Interval={}'
                 .format(num_train_steps, warmup_steps, save_interval))

    lr_scheduler = PolyScheduler(max_update=num_train_steps,
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
    else:
        trainer = mx.gluon.Trainer(param_dict, args.optimizer, optimizer_params,
                                   update_on_kvstore=False)
    if args.start_step:
        logging.info('Restart training from {}'.format(args.start_step))
        # TODO(zheyuye), How about data splitting, where to start re-training
        state_path = states_option(
            args.start_step, trainer, args.output_dir, local_rank, 'Loading')
        param_path = parameters_option(
            args.start_step, model, args.output_dir, 'Loading')

    # prepare the loss function
    mlm_loss_fn = mx.gluon.loss.SoftmaxCELoss()
    rtd_loss_fn = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    mlm_loss_fn.hybridize()
    rtd_loss_fn.hybridize()

    # prepare the records writer
    writer = None
    if args.do_eval and local_rank == 0:
        from tensorboardX import SummaryWriter
        record_path = os.path.join(args.output_dir, 'records')
        logging.info('Evaluation records saved in {}'.format(record_path))
        writer = SummaryWriter(record_path)

    step_num = args.start_step
    finish_flag = False
    num_samples_per_update = 0
    loss_denom = float(len(ctx_l) * args.num_accumulated * num_workers)

    log_total_loss = 0
    log_mlm_loss = 0
    log_rtd_loss = 0
    log_sample_num = 0
    train_start_time = time.time()
    if args.num_accumulated != 1:
        # set grad to zero for gradient accumulation
        model.collect_params().zero_grad()

    # start training
    train_loop_dataloader = grouper(repeat(data_train), len(ctx_l))
    while step_num < num_train_steps:
        tic = time.time()
        for accum_idx in range(args.num_accumulated):
            sample_l = next(train_loop_dataloader)
            loss_l = []
            mlm_loss_l = []
            rtd_loss_l = []
            for sample, ctx in zip(sample_l, ctx_l):
                if sample is None:
                    continue
                # prepare data
                input_ids, segment_ids, valid_lengths = sample
                input_ids = input_ids.as_in_ctx(ctx)
                segment_ids = segment_ids.as_in_ctx(ctx)
                valid_lengths = valid_lengths.as_in_ctx(ctx)
                masked_input = data_masker.dynamic_masking(mx.nd, input_ids, valid_lengths)
                masked_input_ids = masked_input.input_ids
                length_masks = masked_input.masks
                unmasked_tokens = masked_input.unmasked_tokens
                masked_positions = masked_input.masked_positions
                masked_weights = masked_input.masked_weights

                log_sample_num += len(masked_input_ids)
                num_samples_per_update += len(masked_input_ids)

                with mx.autograd.record():
                    mlm_scores, rtd_scores, corrupted_tokens, labels = model(
                        masked_input_ids, segment_ids, valid_lengths, unmasked_tokens, masked_positions)
                    # the official implementation takes the sum of each batch inside the loss function
                    # while SigmoidBinaryCrossEntropyLoss and SoftmaxCELoss takes the mean value
                    mlm_loss = mlm_loss_fn(
                        mlm_scores, unmasked_tokens, masked_weights.reshape(-1)).mean() / (masked_weights.mean() + 1e-6)
                    rtd_loss = rtd_loss_fn(
                        rtd_scores, labels, length_masks).mean() / (length_masks.mean() + 1e-6)
                    output = ElectraOutput(mlm_scores=mlm_scores,
                                           rtd_scores=rtd_scores,
                                           rtd_labels=labels,
                                           corrupted_tokens=corrupted_tokens,
                                           )
                    mlm_loss_l.append(mlm_loss)
                    rtd_loss_l.append(rtd_loss)
                    loss = (args.gen_weight * mlm_loss + args.disc_weight * rtd_loss) / loss_denom
                    loss_l.append(loss)

            for loss in loss_l:
                loss.backward()
            # All Reduce the Step Loss
            log_mlm_loss += sum([ele.as_in_ctx(ctx_l[0])
                                 for ele in mlm_loss_l]).asnumpy()
            log_rtd_loss += sum([ele.as_in_ctx(ctx_l[0])
                                 for ele in rtd_loss_l]).asnumpy()
            log_total_loss += sum([ele.as_in_ctx(ctx_l[0])
                                   for ele in loss_l]).asnumpy() * loss_denom

        # update
        trainer.allreduce_grads()
        # Here, the accumulated gradients are
        # \sum_{n=1}^N g_n / loss_denom
        # Thus, in order to clip the average gradient
        #   \frac{1}{N} \sum_{n=1}^N      -->  clip to args.max_grad_norm
        # We need to change the ratio to be
        #  \sum_{n=1}^N g_n / loss_denom  -->  clip to args.max_grad_norm  * N / loss_denom
        total_norm, ratio, is_finite = clip_grad_global_norm(
            params, args.max_grad_norm * num_samples_per_update / loss_denom)
        total_norm = total_norm / (num_samples_per_update / loss_denom)
        trainer.update(num_samples_per_update / loss_denom, ignore_stale_grad=True)
        step_num += 1
        if args.num_accumulated != 1:
            # set grad to zero for gradient accumulation
            model.collect_params().zero_grad()

        # saving
        if step_num % save_interval == 0 or step_num >= num_train_steps:
            if is_master_node:
                states_option(
                    step_num, trainer, args.output_dir, local_rank, 'Saving')
                if local_rank == 0:
                    param_path = parameters_option(
                        step_num, model, args.output_dir, 'Saving')

        # logging
        if step_num % log_interval == 0 and local_rank == 0:
            # Output the loss of per step
            log_mlm_loss /= log_interval
            log_rtd_loss /= log_interval
            log_total_loss /= log_interval
            toc = time.time()
            logging.info(
                '[step {}], Loss mlm/rtd/total={:.4f}/{:.4f}/{:.4f},'
                ' LR={:.6f}, grad_norm={:.4f}. Time cost={:.2f},'
                ' Throughput={:.2f} samples/s, ETA={:.2f}h'.format(
                    step_num, log_mlm_loss, log_rtd_loss, log_total_loss,
                    trainer.learning_rate, total_norm, toc - tic, log_sample_num / (toc - tic),
                    (num_train_steps - step_num) / (step_num / (toc - train_start_time)) / 3600))
            tic = time.time()

            if args.do_eval:
                evaluation(writer, step_num, masked_input, output)
                writer.add_scalars('loss',
                                   {'total_loss': log_total_loss,
                                    'mlm_loss': log_mlm_loss,
                                    'rtd_loss': log_rtd_loss},
                                   step_num)
            log_mlm_loss = 0
            log_rtd_loss = 0
            log_total_loss = 0
            log_sample_num = 0

        num_samples_per_update = 0

    logging.info('Finish training step: %d', step_num)
    if is_master_node:
        state_path = states_option(step_num, trainer, args.output_dir, local_rank, 'Saving')
        if local_rank == 0:
            param_path = parameters_option(step_num, model, args.output_dir, 'Saving')

    mx.npx.waitall()
    train_end_time = time.time()
    logging.info('Train cost={:.1f}s'.format(
        train_end_time - train_start_time))
    if writer is not None:
        writer.close()
        
    if local_rank == 0:
        model_name = args.model_name.replace('google', 'gluon')
        save_dir = os.path.join(args.output_dir, model_name)
        final_save(model, save_dir, tokenizer)

# TODO(zheyuye), Directly implement a metric for weighted accuracy


def accuracy(labels, predictions, weights=None):
    if weights is None:
        weights = mx.np.ones_like(labels)
    is_correct = mx.np.equal(labels, predictions)
    acc = (is_correct * weights).sum() / (weights.sum() + 1e-6)
    return acc

# TODO(zheyuye), Directly implement a metric for weighted AUC


def auc(labels, probs, weights=None):
    if isinstance(labels, mx.np.ndarray):
        labels = labels.asnumpy()
    if isinstance(probs, mx.np.ndarray):
        probs = probs.asnumpy()
    if isinstance(weights, mx.np.ndarray):
        weights = weights.asnumpy()
    labels = labels.reshape(-1)
    probs = probs.reshape(-1)
    weights = weights.reshape(-1)

    fpr, tpr, thresholds = metrics.roc_curve(labels, probs, sample_weight=weights)
    return metrics.auc(fpr, tpr)


def evaluation(writer, step_num, masked_input, eval_input):
    length_masks = masked_input.masks
    unmasked_tokens = masked_input.unmasked_tokens
    masked_weights = masked_input.masked_weights
    mlm_scores = eval_input.mlm_scores
    rtd_scores = eval_input.rtd_scores
    rtd_labels = eval_input.rtd_labels
    corrupted_tokens = eval_input.corrupted_tokens

    mlm_log_probs = mx.npx.log_softmax(mlm_scores)
    mlm_preds = mx.np.argmax(mlm_log_probs, axis=-1).astype(np.int32)
    rtd_probs = mx.npx.sigmoid(rtd_scores)
    rtd_preds = mx.np.round((mx.np.sign(rtd_scores) + 1) / 2).astype(np.int32)

    mlm_accuracy = accuracy(unmasked_tokens, mlm_preds, masked_weights)
    corrupted_mlm_accuracy = accuracy(unmasked_tokens, corrupted_tokens, masked_weights)
    rtd_accuracy = accuracy(rtd_labels, rtd_preds, length_masks)
    rtd_precision = accuracy(rtd_labels, rtd_preds, length_masks * rtd_preds)
    rtd_recall = accuracy(rtd_labels, rtd_preds, rtd_labels * rtd_preds)
    rtd_auc = auc(rtd_labels, rtd_probs, length_masks)
    writer.add_scalars('results',
                       {'mlm_accuracy': mlm_accuracy.asnumpy().item(),
                        'corrupted_mlm_accuracy': corrupted_mlm_accuracy.asnumpy().item(),
                        'rtd_accuracy': rtd_accuracy.asnumpy().item(),
                        'rtd_precision': rtd_precision.asnumpy().item(),
                        'rtd_recall': rtd_recall.asnumpy().item(),
                        'rtd_auc': rtd_auc},
                       step_num)


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    os.environ['MXNET_USE_FUSION'] = '0'  # Manually disable pointwise fusion
    args = parse_args()
    logging_config(args.output_dir, name='pretrain_owt')
    logging.debug('Random seed set to {}'.format(args.seed))
    logging.info(args)
    set_seed(args.seed)
    if args.do_train:
        train(args)
