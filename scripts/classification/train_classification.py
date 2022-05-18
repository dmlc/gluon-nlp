import gluonnlp
from tensorboardX import SummaryWriter
import numpy as np
import mxnet as mx
import json
import random
import pandas as pd
import os
import logging
import time
import argparse
import copy
from mxnet.gluon.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
from classification_utils import get_task
import matplotlib.pyplot as plt
from tqdm import tqdm
from mxnet import gluon
from gluonnlp.data.sampler import SplitSampler
from mxnet.gluon import nn
from gluonnlp.models import get_backbone
from gluonnlp.utils.parameter import clip_grad_global_norm, count_parameters, deduplicate_param_dict
from gluonnlp.utils.preprocessing import get_trimmed_lengths
from gluonnlp.utils.misc import get_mxnet_visible_device, grouper, repeat, logging_config
from mxnet.gluon.data import batchify as bf
from mxnet.gluon.data import DataLoader
from mxnet.lr_scheduler import PolyScheduler
from gluonnlp.utils import set_seed
from gluonnlp.utils.misc import init_comm, parse_device
try:
    import horovod.mxnet as hvd
except ImportError:
    pass
from classification import TextPredictionNet



CACHE_PATH = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', 'cached'))
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='classification example. '
                    'We fine-tune the pretrained model on GLUE dataset to do different taks.')
    parser.add_argument('--model_name', type=str, default='google_en_uncased_bert_base',
                        help='Name of the pretrained model.')
    parser.add_argument('--task_name', type=str, default='STS',
                        help='Name of classification taks')
    parser.add_argument('--lr', type=float, default=5E-4,
                        help='Initial learning rate. default is 2e-5')
    parser.add_argument('--comm_backend', type=str, default='device',
                        choices=['horovod', 'dist_sync_device', 'device'],
                        help='Communication backend.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs, default is 3')
    parser.add_argument('--do_train', action='store_true',
                        help='do training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='do eval.')
    parser.add_argument('--param_checkpoint', type=str, default=None,
                        help='The parameter checkpoint for evaluating the model')
    parser.add_argument('--backbone_path', type=str, default=None,
                        help='The parameter checkpoint of backbone model')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Whether to overwrite the feature cache.')
    parser.add_argument('--num_accumulated', type=int, default=1,
                        help='The number of batches for gradients accumulation to '
                             'simulate large batch size.')
    parser.add_argument('--output_dir', type=str, default='cls_dir',
                        help='The output directory where the model params will be written.'
                             ' default is cls_dir')
    parser.add_argument('--log_interval', type=int, default=-1,
                        help='The logging interval for training')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='The optimization algorithm')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size. Number of examples per gpu in a minibatch. default is 64')
    parser.add_argument(
        '--seed', type=int, default=2, help='Random seed')

    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm.')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='the path to training dataset')
    parser.add_argument('--eval_dir', type=str, default=None,
                        help='the path to training dataset')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Ratio of warmup steps in the learning rate scheduler.')


    args = parser.parse_args()
    return args

def get_network(model_name,
                device_l,
                checkpoint_path=None,
                backbone_path=None,
                task=None):
    """
    Get the network that fine-tune the Question Answering Task
    """

    use_segmentation = 'roberta' not in model_name and 'xlmr' not in model_name
    Model, cfg, tokenizer, download_params_path, _ = \
        get_backbone(model_name, load_backbone=not backbone_path)
    backbone = Model.from_cfg(cfg)
    # Load local backbone parameters if backbone_path provided.
    # Otherwise, download backbone parameters from gluon zoo.

    backbone_params_path = backbone_path if backbone_path else download_params_path
    if checkpoint_path is None:
        backbone.load_parameters(backbone_params_path, ignore_extra=True,
                                 device=device_l, cast_dtype=True)
        num_params, num_fixed_params \
            = count_parameters(deduplicate_param_dict(backbone.collect_params()))
        logging.info(
            'Loading Backbone Model from {}, with total/fixd parameters={}/{}'.format(
                backbone_params_path, num_params, num_fixed_params))
    classify_net = TextPredictionNet(backbone, task.class_num)
    if checkpoint_path is None:
        # Ignore the UserWarning during initialization,
        # There is no need to re-initialize the parameters of backbone
        classify_net.initialize(device=device_l)
    else:
        classify_net.load_parameters(checkpoint_path, device=device_l, cast_dtype=True)
    classify_net.hybridize()

    return cfg, tokenizer, classify_net, use_segmentation

def project_label(label, task):
    projected_label = copy.copy(label)
    for i in range(len(label)):
        projected_label[i] = task.proj_label[label[i]]

    return projected_label



def preprocess_data(df, feature_columns, label_column, tokenizer,
                    max_length=128, use_label=True, use_tqdm=True, task=None):
    out = []
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    cls_id = tokenizer.vocab.cls_id
    sep_id = tokenizer.vocab.sep_id
    iterator = tqdm(df.iterrows(), total=len(df)) if use_tqdm else df.iterrows()
    for idx, row in iterator:
        # Token IDs =      [CLS]    token_ids1       [SEP]      token_ids2         [SEP]
        # Segment IDs =      0         0               0           1                 1

        encoded_text_l = [tokenizer.encode(row[col_name], int)
                          for col_name in feature_columns]
        trimmed_lengths = get_trimmed_lengths([len(ele) for ele in encoded_text_l],
                                              max_length=max_length - len(feature_columns) - 1,
                                              do_merge=True)

        token_ids = [cls_id] + sum([ele[:length] + [sep_id]
                          for length, ele in zip(trimmed_lengths, encoded_text_l)], [])
        token_types = [0] + sum([[i % 2] * (length + 1)
                                 for i, length in enumerate(trimmed_lengths)], [])
        valid_length = len(token_ids)
        feature = (token_ids, token_types, valid_length)
        if use_label:
            label = row[label_column]
            if task.task_name != 'sts':
                label = task.proj_label[label]
            out.append((feature, label))
        else:
            out.append(feature)

    return out


def get_task_data(args, task, tokenizer, segment):
    feature_column = task.feature_column
    label_column = task.label_column
    if segment == 'train':
        input_df = task.raw_train_data
        file_name = args.train_dir.split('/')[-1]
    else:
        input_df = task.raw_eval_data
        file_name = args.eval_dir.split('/')[-1]
    data_cache_path = os.path.join(CACHE_PATH,
                                   '{}_{}_{}_{}.ndjson'.format(
                                       segment, args.model_name, task.task_name, file_name))
    if os.path.exists(data_cache_path) and not args.overwrite_cache:
        processed_data = []
        with open(data_cache_path, 'r') as f:
            for line in f:
                processed_data.append(json.loads(line))
        logging.info('Found cached data features, load from {}'.format(data_cache_path))
    else:
        processed_data = preprocess_data(input_df, feature_column, label_column,
                                         tokenizer, use_label=True, task=task)
        with open(data_cache_path, 'w') as f:
            for feature in processed_data:
                f.write(json.dumps(feature) + '\n')

    label = input_df[label_column]
    if task.task_name != 'sts':
        label = project_label(label, task)
    return processed_data, label





def train(args):
    store, num_workers, rank, local_rank, is_master_node, device_l = init_comm(
        args.comm_backend, args.gpus)
    task = get_task(args.task_name, args.train_dir, args.eval_dir)
    #setup_logging(args, local_rank)
    #random seed
    set_seed(args.seed)
    level = logging.INFO
    detail_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(detail_dir):
        os.mkdir(detail_dir)
    logging_config(detail_dir,
                   name='train_{}_{}_'.format(args.task_name, args.model_name) + str(rank),  # avoid race
                   level=level,
                   console=(local_rank == 0))
    logging.info(args)
    cfg, tokenizer, classify_net, use_segmentation = \
        get_network(args.model_name, device_l,
                    args.param_checkpoint,
                    args.backbone_path,
                    task)

    logging.info('Prepare training data')
    train_data, _ = get_task_data(args, task, tokenizer, segment='train')
    train_batchify = bf.Group(bf.Group(bf.Pad(), bf.Pad(), bf.Stack()),
                              bf.Stack())

    rs = np.random.RandomState(100)
    rs.shuffle(train_data)
    sampler = SplitSampler(
        len(train_data),
        num_parts=num_workers,
        part_index=rank,
        even_size=True)

    dataloader = DataLoader(train_data,
                            batch_size=args.batch_size,
                            batchify_fn=train_batchify,
                            num_workers=0,
                            sampler=sampler)



    param_dict = classify_net.collect_params()
    # Do not apply weight decay to all the LayerNorm and bias
    for _, v in classify_net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Set grad_req if gradient accumulation is required
    params = [p for p in param_dict.values() if p.grad_req != 'null']
    num_accumulated = args.num_accumulated
    if num_accumulated > 1:
        logging.info('Using gradient accumulation. Effective global batch size = {}'
                     .format(num_accumulated * args.batch_size * len(device_l) * num_workers))
        for p in params:
            p.grad_req = 'add'
    if local_rank == 0:
        writer = SummaryWriter(logdir=os.path.join(args.output_dir,
                                                   args.task_name + '_tensorboard_' +
                                                   str(args.lr) + '_' + str(args.epochs)))
    if args.comm_backend == 'horovod':
        # Horovod: fetch and broadcast parameters
        hvd.broadcast_parameters(param_dict, root_rank=0)

    epoch_size = (len(dataloader) + len(device_l) - 1) // len(device_l)
    max_update = epoch_size * args.epochs
    warmup_steps = int(np.ceil(max_update * args.warmup_ratio))

    dataloader = grouper(repeat(dataloader), len(device_l))

    lr_scheduler = PolyScheduler(max_update=max_update,
                                 base_lr=args.lr,
                                 warmup_begin_lr=0.0,
                                 pwr=1,
                                 final_lr=0.0,
                                 warmup_steps=warmup_steps,
                                 warmup_mode='linear')
    optimizer_params = {'learning_rate': args.lr,
                        'wd': args.wd,
                        'lr_scheduler': lr_scheduler}
    if args.comm_backend == 'horovod':
        trainer = hvd.DistributedTrainer(param_dict, args.optimizer, optimizer_params)
    else:
        trainer = mx.gluon.Trainer(classify_net.collect_params(),
                                   'adamw',
                                   optimizer_params)

    if args.task_name == 'sts':
        loss_function = gluon.loss.L2Loss()
    else:
        loss_function = gluon.loss.SoftmaxCELoss()

    metrics = task.metric
    #prepare loss function
    log_loss = 0
    log_gnorm = 0
    log_step = 0
    if args.log_interval > 0:
        log_interval = args.log_interval
    else:
        log_interval = int(epoch_size * 0.5)

    start_time = time.time()
    total_loss = 0
    total_grad = 0
    total_step = 0
    for i in range(max_update):
        sample_l = next(dataloader)
        loss_l = []
        for sample, device in zip(sample_l, device_l):
            (token_ids, token_types, valid_length), label = sample
            # Move to the corresponding context
            token_ids = mx.np.array(token_ids, device=device)
            token_types = mx.np.array(token_types, device=device)
            valid_length = mx.np.array(valid_length, device=device)
            label = mx.np.array(label, device=device)
            with mx.autograd.record():
                scores = classify_net(token_ids, token_types, valid_length)
                loss = loss_function(scores, label).mean() / len(device_l)
                loss_l.append(loss)
            if task.task_name == 'sts':
                label = label.reshape((-1, 1))
            for metric in metrics:
                metric.update([label], [scores])

        for loss in loss_l:
            loss.backward()
        trainer.allreduce_grads()
        # Begin Norm Clipping
        total_norm, ratio, is_finite = clip_grad_global_norm(params, args.max_grad_norm)
        trainer.update(1.0)
        step_loss = sum([loss.asnumpy() for loss in loss_l])
        log_loss += step_loss
        log_gnorm += total_norm
        log_step += 1
        total_step += 1
        total_loss += step_loss
        total_grad += total_norm
        if local_rank == 0:
            writer.add_scalar('train_loss_avg', total_loss * 1.0 / total_step, i)
            writer.add_scalar('lr', trainer.learning_rate, i)
            writer.add_scalar('train_loss', step_loss, i)
            writer.add_scalar('grad_norm_avg', total_grad * 1.0 / total_step, i)
            writer.add_scalar('grad_norm', total_norm, i)
            for metric in metrics:
                metric_name, result = metric.get()
                writer.add_scalar(metric_name, result, i)
        if log_step >= log_interval or i == max_update - 1:
            curr_time = time.time()
            metric_log = ''
            for metric in metrics:
                metric_nm, val = metric.get()
                metric_log += ', {}: = {}'.format(metric_nm, val)
            logging.info('[Iter {} / {}] avg {} = {:.2f}, avg gradient norm = {:.2f}, lr = {}, ETA={:.2f}h'.format(i + 1,
                                                                                      max_update,
                                                                                      'loss',
                                                                                      log_loss / log_step,
                                                                                      log_gnorm / log_step,
                                                                                      trainer.learning_rate,

                                                                         (max_update-i)*((curr_time - start_time)/i)/3600)
                                                                                + metric_log)
            log_loss = 0
            log_gnorm = 0
            log_step = 0
        if local_rank == 0 and (i == max_update - 1 or i%(max_update//args.epochs) == 0 and i>0):
            ckpt_name = '{}_{}_{}.params'.format(args.model_name,
                                                 args.task_name,
                                                 (i + 1))

            params_saved = os.path.join(detail_dir, ckpt_name)
            classify_net.save_parameters(params_saved)
            logging.info('Params saved in: {}'.format(params_saved))
            for metric in metrics:
                metric.reset()



def evaluate(args):
    store, num_workers, rank, local_rank, is_master_node, device_l = init_comm(
        args.comm_backend, args.gpus)
    # setup_logging(args, local_rank)
    task = get_task(args.task_name, args.train_dir, args.eval_dir)
    level = logging.INFO
    detail_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(detail_dir):
        os.mkdir(detail_dir)
    logging_config(detail_dir,
                   name='train_{}_{}_'.format(args.task_name, args.model_name) + str(rank),  # avoid race
                   level=level,
                   console=(local_rank == 0))
    if rank != 0:
        logging.info('Skipping node {}'.format(rank))
        return
    device_l = parse_device(args.gpus)
    logging.info(
        'Srarting inference without horovod on the first node on device {}'.format(
            str(device_l)))

    cfg, tokenizer, classify_net, use_segmentation = \
        get_network(args.model_name, device_l,
                    args.param_checkpoint,
                    args.backbone_path,
                    task)
    candidate_ckpt = []
    detail_dir = os.path.join(args.output_dir, args.task_name)
    for name in os.listdir(detail_dir):
        if name.endswith('.params') and args.task_name in name and args.model_name in name:
            candidate_ckpt.append(os.path.join(detail_dir, name))
    best_ckpt = {}
    metrics = task.metric
    def evaluate_by_ckpt(ckpt_name, best_ckpt):
        classify_net.load_parameters(ckpt_name, device=device_l, cast_dtype=True)
        logging.info('Prepare dev data')

        dev_data, label = get_task_data(args, task, tokenizer, segment='eval')
        dev_batchify = bf.Group(bf.Group(bf.Pad(), bf.Pad(), bf.Stack()), bf.Stack())
        dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                batchify_fn=dev_batchify,
                                shuffle=False)

        for sample_l in grouper(dataloader, len(device_l)):
            for sample, device in zip(sample_l, device_l):
                if sample is None:
                    continue
                (token_ids, token_types, valid_length), label = sample
                token_ids = mx.np.array(token_ids, device=device)
                token_types = mx.np.array(token_types, device=device)
                valid_length = mx.np.array(valid_length, device=device)
                scores = classify_net(token_ids, token_types, valid_length)

                if task.task_name == 'sts':
                    label = label.reshape((-1,1))
                for metric in metrics:
                    metric.update([label], [scores])
                #pred.append(scores)


        for metric in metrics:
            metric_name, result = metric.get()
            logging.info('checkpoint {} get result: {}:{}'.format(ckpt_name, metric_name, result))
            if best_ckpt.get(metric_name, [0, ''])[0]<result:
                best_ckpt[metric_name] = [result, ckpt_name]
        for metric in metrics:
            metric.reset()

    for ckpt_name in candidate_ckpt:
        evaluate_by_ckpt(ckpt_name, best_ckpt)
    for metric_name in best_ckpt:
        logging.info('best result on metric {}: is {}, and on checkpoint {}'.format(metric_name, best_ckpt[metric_name][0],
                                                                                    best_ckpt[metric_name][1]))







if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    args = parse_args()
    if args.do_train:
        train(args)
    if args.do_eval:
        evaluate(args)
