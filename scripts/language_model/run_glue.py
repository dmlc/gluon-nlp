"""
Sentence Pair Classification with XLNet
"""
import io
import os
import time
import argparse
import random
import logging
import warnings
import sys
from functools import partial
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from model.XLNet_classifier import XLNetClassifier
from transformer import model

path = sys.path[0]
sys.path.append(path + '/../bert/data')
#pylint: disable=wrong-import-position
from classification import MRPCTask, QQPTask, RTETask, STSBTask, SSTTask, \
     QNLITask, CoLATask, MNLITask, WNLITask, XNLITask, LCQMCTask, ChnSentiCorpTask
from preprocessing_utils import truncate_seqs_equal, concat_sequences

tasks = {
    'MRPC': MRPCTask(),
    'QQP': QQPTask(),
    'QNLI': QNLITask(),
    'RTE': RTETask(),
    'STS-B': STSBTask(),
    'CoLA': CoLATask(),
    'MNLI': MNLITask(),
    'WNLI': WNLITask(),
    'SST': SSTTask(),
    'XNLI': XNLITask(),
    'LCQMC': LCQMCTask(),
    'ChnSentiCorp': ChnSentiCorpTask()
}

parser = argparse.ArgumentParser(
    description='XLNet fine-tune examples for classification/regression tasks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Training config
parser.add_argument('--epochs', type=int, default=3, help='number of epochs.')
parser.add_argument('--training_steps',
                    type=int,
                    help='If specified, epochs will be ignored.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size. Number of examples per gpu in a minibatch.')

parser.add_argument(
    '--accumulate',
    type=int,
    default=None,
    help=
    'The number of batches for gradients accumulation to simulate large batch size. '
    'Default is None')

parser.add_argument('--dev_batch_size',
                    type=int,
                    default=32,
                    help='Batch size for dev set and test set')

parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--attention_dropout',
                    type=float,
                    default=0.1,
                    help='attention dropout')
parser.add_argument('--log_interval',
                    type=int,
                    default=10,
                    help='report interval')
parser.add_argument(
    '--early_stop',
    type=int,
    default=None,
    help='Whether to perform early stopping based on the metric on dev set. '
    'The provided value is the patience. ')

# Optimizer config
parser.add_argument('--optimizer', type=str, default='Adam', help='')
parser.add_argument('--lr',
                    type=float,
                    default=3e-5,
                    help='Initial learning rate')
parser.add_argument('--lr_decay',
                    type=str,
                    choices=['linear'],
                    default='linear',
                    help='lr schedule')
parser.add_argument('--epsilon',
                    type=float,
                    default=1e-6,
                    help='Small value to avoid division by 0')
parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0,
    help='ratio of warmup steps used in NOAM\'s stepsize schedule')

# task spesific & data preprocessing
parser.add_argument('--gpu',
                    type=int,
                    default=None,
                    help='Number of gpus for finetuning.')
parser.add_argument('--task_name',
                    default='MRPC',
                    type=str,
                    help='The name of the task to fine-tune.')

parser.add_argument(
    '--model_name',
    type=str,
    default='xlnet_cased_l12_h768_a12',
    choices=['xlnet_cased_l24_h1024_a16', 'xlnet_cased_l12_h768_a12'],
    help='The name of pre-trained XLNet model to fine-tune')

parser.add_argument('--dataset',
                    type=str,
                    default='126gb',
                    help='The dataset BERT pre-trained with.')
parser.add_argument('--max_len',
                    type=int,
                    default=128,
                    help='Maximum length of the sentence pairs')

parser.add_argument(
    '--round_to', type=int, default=None,
    help='The length of padded sequences will be rounded up to be multiple of this argument.'
         'When round to is set to 8, training throughput may increase for mixed precision'
         'training on GPUs with tensorcores.')

parser.add_argument(
    '--only_inference',
    action='store_true',
    help=
    'If set, we skip training and only perform inference on dev and test data.'
)

# Initializing config
parser.add_argument('--seed', type=int, default=2, help='Random seed')

# I/O config
parser.add_argument(
    '--output_dir',
    type=str,
    default='./output_dir',
    help='The output directory where the model params will be written.')
parser.add_argument(
    '--model_parameters',
    type=str,
    default=None,
    help='A parameter file for the model that is loaded into the model'
    ' before training/inference. It is different from the parameter'
    ' file written after the model is trained.')

args = parser.parse_args()


def split_array(arr, num_of_splits):
    """split an array into equal pieces"""
    # TODO Replace this function with gluon.utils.split_data() once targeting MXNet 1.7
    size = arr.shape[0]
    if size < num_of_splits:
        return [arr[i:i + 1] for i in range(size)]
    slice_len, rest = divmod(size, num_of_splits)
    div_points = [0] + [(slice_len * index + min(index, rest) + slice_len +
                         (index < rest)) for index in range(num_of_splits)]
    slices = [
        arr[div_points[i]:div_points[i + 1]] for i in range(num_of_splits)
    ]
    return slices


def split_and_load(arrs, _ctxs):
    """split and load arrays to a list of contexts"""
    # TODO Replace split_array() with gluon.utils.split_data() once targeting MXNet 1.7
    assert isinstance(arrs, (list, tuple))
    # split and load
    loaded_arrs = [[
        i.as_in_context(ctx)
        for i, ctx in zip(split_array(arr, len(_ctxs)), _ctxs)
    ] for arr in arrs]
    return zip(*loaded_arrs)


def convert_examples_to_features(example,
                                 tokenizer=None,
                                 truncate_length=512,
                                 cls_token=None,
                                 sep_token=None,
                                 class_labels=None,
                                 label_alias=None,
                                 vocab=None,
                                 is_test=False):
    #pylint: disable=redefined-outer-name
    """convert glue examples into necessary features"""
    assert vocab
    if not is_test:
        label_dtype = 'int32' if class_labels else 'float32'
        # get the label
        label = example[-1]
        example = example[:-1]
        #create label maps if classification task
        if class_labels:
            label_map = {}
            for (i, l) in enumerate(class_labels):
                label_map[l] = i
            if label_alias:
                for key in label_alias:
                    label_map[key] = label_map[label_alias[key]]
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = truncate_seqs_equal(tokens_raw, truncate_length)
    # concate the sequences with special tokens, cls_token is added to the end in XlNet
    special_tokens = [[sep_token]] * len(tokens_trun) + [[cls_token]]
    tokens, segment_ids, _ = concat_sequences(tokens_trun, special_tokens)
    # convert the token to ids
    input_ids = vocab[tokens]
    valid_length = len(input_ids)
    if not is_test:
        return input_ids, valid_length, segment_ids, label
    else:
        return input_ids, valid_length, segment_ids


def preprocess_data(_tokenizer,
                    _task,
                    batch_size,
                    dev_batch_size,
                    max_len,
                    _vocab):
    """Train/eval Data preparation function."""
    label_dtype = 'int32' if _task.class_labels else 'float32'
    truncate_length = max_len - 3 if _task.is_pair else max_len - 2
    trans = partial(convert_examples_to_features,
                    tokenizer=_tokenizer,
                    truncate_length=truncate_length,
                    cls_token=_vocab.cls_token,
                    sep_token=_vocab.sep_token,
                    class_labels=_task.class_labels,
                    label_alias=_task.label_alias,
                    vocab=_vocab)

    # data train
    # task.dataset_train returns (segment_name, dataset)
    train_tsv = _task.dataset_train()[1]
    data_train = list(map(trans, train_tsv))
    data_train = mx.gluon.data.SimpleDataset(data_train)
    data_train_len = data_train.transform(
        lambda _, valid_length, segment_ids, label: valid_length, lazy=False)

    # bucket sampler for training
    pad_val = _vocab[_vocab.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val, round_to=args.round_to),  # input
        nlp.data.batchify.Stack(),  # length
        nlp.data.batchify.Pad(axis=0, pad_val=4, round_to=args.round_to),  # segment
        nlp.data.batchify.Stack(label_dtype))  # label
    batch_sampler = nlp.data.sampler.FixedBucketSampler(data_train_len,
                                                        batch_size=batch_size,
                                                        num_buckets=10,
                                                        ratio=0,
                                                        shuffle=True)
    # data loader for training
    loader_train = gluon.data.DataLoader(dataset=data_train,
                                         num_workers=4,
                                         batch_sampler=batch_sampler,
                                         batchify_fn=batchify_fn)

    # data dev. For MNLI, more than one dev set is available
    dev_tsv = _task.dataset_dev()
    dev_tsv_list = dev_tsv if isinstance(dev_tsv, list) else [dev_tsv]
    loader_dev_list = []
    for segment, data in dev_tsv_list:
        data_dev = mx.gluon.data.SimpleDataset(list(map(trans, data)))
        loader_dev = mx.gluon.data.DataLoader(data_dev,
                                              batch_size=dev_batch_size,
                                              num_workers=4,
                                              shuffle=False,
                                              batchify_fn=batchify_fn)
        loader_dev_list.append((segment, loader_dev))

    # batchify for data test
    test_batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val, round_to=args.round_to),
        nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=0, round_to=args.round_to))

    # transform for data test
    test_trans = partial(convert_examples_to_features,
                         tokenizer=_tokenizer,
                         truncate_length=max_len,
                         cls_token=_vocab.cls_token,
                         sep_token=_vocab.sep_token,
                         class_labels=None,
                         is_test=True,
                         vocab=_vocab)

    # data test. For MNLI, more than one test set is available
    test_tsv = _task.dataset_test()
    test_tsv_list = test_tsv if isinstance(test_tsv, list) else [test_tsv]
    loader_test_list = []
    for segment, data in test_tsv_list:
        data_test = mx.gluon.data.SimpleDataset(list(map(test_trans, data)))
        loader_test = mx.gluon.data.DataLoader(data_test,
                                               batch_size=dev_batch_size,
                                               num_workers=4,
                                               shuffle=False,
                                               batchify_fn=test_batchify_fn)
        loader_test_list.append((segment, loader_test))
    return loader_train, loader_dev_list, loader_test_list, len(data_train)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.captureWarnings(True)
handler = logging.FileHandler('log_{0}.txt'.format(args.task_name))
handler.setLevel(logging.INFO)
handler2 = logging.StreamHandler()
handler2.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler2.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(handler2)
logging.info(args)

log_interval = args.log_interval * args.accumulate if args.accumulate else args.log_interval

if args.accumulate:
    logging.info('Using gradient accumulation. Effective batch size = ' \
                 'batch_size * accumulate = %d', args.accumulate * args.batch_size)

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)

num_workers = 0
ctxs = [mx.cpu(0)] if not args.gpu else [mx.gpu(i) for i in range(args.gpu)]

task = tasks[args.task_name]

# model and loss
if args.only_inference and not args.model_parameters:
    warnings.warn('model_parameters is not set. '
                  'Randomly initialized model will be used for inference.')

get_pretrained = True

get_model_params = {
    'name': args.model_name,
    'dataset_name': args.dataset,
    'pretrained': get_pretrained,
    'ctx': ctxs,
    'use_decoder': False,
    'dropout': args.dropout,
    'attention_dropout': args.attention_dropout
}

xlnet_base, vocab, tokenizer = model.get_model(**get_model_params)
# initialize the rest of the parameters
initializer = mx.init.Normal(0.02)

do_regression = not task.class_labels
if do_regression:
    num_classes = 1
    loss_function = gluon.loss.L2Loss()
else:
    num_classes = len(task.class_labels)
    loss_function = gluon.loss.SoftmaxCELoss()
# reuse the XLnetClassifier class with num_classes=1 for regression
model = XLNetClassifier(xlnet_base,
                        units=xlnet_base._net._units,
                        dropout=0.1,
                        num_classes=num_classes)

num_ctxes = len(ctxs)

# initialize classifier
if not args.model_parameters:
    model.classifier.initialize(init=initializer, ctx=ctxs)
    model.pooler.initialize(init=initializer, ctx=ctxs)

# load checkpointing
output_dir = args.output_dir

if args.model_parameters:
    logging.info('loading model params from %s', args.model_parameters)
    nlp.utils.load_parameters(model,
                              args.model_parameters,
                              ctx=ctxs,
                              cast_dtype=True)

nlp.utils.mkdir(output_dir)

logging.debug(model)
model.hybridize(static_alloc=True)
loss_function.hybridize(static_alloc=True)

logging.info('processing dataset...')
train_data, dev_data_list, test_data_list, num_train_examples = preprocess_data(
    tokenizer, task, args.batch_size, args.dev_batch_size, args.max_len, vocab)


def test(loader_test, segment):
    """Inference function on the test dataset."""
    logging.info('Now we are doing testing on %s with %s.', segment, ctxs)

    tic = time.time()
    results = []
    for _, seqs in enumerate(loader_test):
        #input_ids, valid_length, segment_ids = seqs
        data_list = list(split_and_load(seqs, ctxs))
        out_list = []
        for splited_data in data_list:
            input_ids, valid_length, segment_ids = splited_data
            out = model(input_ids, segment_ids, valid_length=valid_length)
            out_list.append(out)
        out_list = np.vstack([o.asnumpy() for o in out_list])
        if not task.class_labels:
            # regression task
            for result in out_list.reshape(-1).tolist():
                results.append('{:.3f}'.format(result))
        else:
            # classification task
            out = out_list.reshape(-1, out_list.shape[-1])
            indices = out.argmax(axis=-1)
            for index in indices:
                results.append(task.class_labels[int(index)])

    mx.nd.waitall()
    toc = time.time()
    logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                 args.dev_batch_size * len(loader_test) / (toc - tic))
    # write result to a file.
    segment = segment.replace('_mismatched', '-mm')
    segment = segment.replace('_matched', '-m')
    segment = segment.replace('SST', 'SST-2')
    filename = args.task_name + segment.replace('test', '') + '.tsv'
    test_path = os.path.join(args.output_dir, filename)
    with io.open(test_path, 'w', encoding='utf-8') as f:
        f.write(u'index\tprediction\n')
        for i, pred in enumerate(results):
            f.write(u'%d\t%s\n' % (i, str(pred)))


def log_metric(metric, is_training=True):
    prefix = 'training' if is_training else 'validation'
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]
    logging_str = prefix + ' metrics:' + ','.join(
        [i + ':%.4f' for i in metric_nm])
    logging.info(logging_str, *metric_val)
    return metric_nm, metric_val


def log_train(batch_id, batch_num, step_loss, _log_interval, epoch_id,
              learning_rate):
    """Generate and print out the log message for training. """
    train_str = '[Epoch %d Batch %d/%d] loss=%.4f, lr=%.7f'
    logging.info(train_str, epoch_id + 1, batch_id + 1, batch_num,
                 step_loss / _log_interval, learning_rate)


def log_eval(batch_id, batch_num, step_loss, _log_interval):
    """Generate and print out the log message for inference. """
    eval_str = '[Batch %d/%d] loss=%.4f'
    logging.info(eval_str, batch_id + 1, batch_num, step_loss / _log_interval)


def train(metric):
    """Training function."""
    if not args.only_inference:
        logging.info('Now we are doing XLNet classification training on %s!',
                     ctxs)

    all_model_params = model.collect_params()
    optimizer_params = {
        'learning_rate': args.lr,
        'epsilon': args.epsilon,
        'wd': 0
    }
    trainer = gluon.Trainer(all_model_params,
                            args.optimizer,
                            optimizer_params,
                            update_on_kvstore=False)

    step_size = args.batch_size * args.accumulate if args.accumulate else args.batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    epoch_number = args.epochs
    if args.training_steps:
        num_train_steps = args.training_steps
        epoch_number = 9999
    logging.info('training steps=%d', num_train_steps)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in all_model_params.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if args.accumulate and args.accumulate > 1:
        for p in params:
            p.grad_req = 'add'
    # track best eval score
    metric_history = []
    best_metric = None
    patience = args.early_stop

    tic = time.time()
    finish_flag = False
    for epoch_id in range(epoch_number):
        if args.early_stop and patience == 0:
            logging.info('Early stopping at epoch %d', epoch_id)
            break
        if finish_flag:
            break
        if not args.only_inference:
            metric.reset()
            step_loss = 0
            tic = time.time()
            all_model_params.zero_grad()
            for batch_id, seqs in enumerate(train_data):
                new_lr = args.lr
                # learning rate schedule
                if step_num < num_warmup_steps:
                    new_lr = args.lr * step_num / num_warmup_steps
                elif args.lr_decay == 'linear':
                    non_warmup_steps = step_num - num_warmup_steps
                    offset = non_warmup_steps / (num_train_steps -
                                                 num_warmup_steps)
                    new_lr = max(0, args.lr - offset * args.lr)
                trainer.set_learning_rate(new_lr)
                batch_loss = []
                # forward and backward
                with mx.autograd.record():
                    data_list = list(split_and_load(seqs, ctxs))
                    for splited_data in data_list:
                        input_ids, valid_length, segment_ids, label = splited_data
                        out = model(input_ids,
                                    segment_ids,
                                    valid_length=valid_length)
                        ls = loss_function(out, label).mean() / len(ctxs)
                        batch_loss.append(ls)
                        if args.accumulate:
                            ls = ls / args.accumulate
                        ls.backward()
                # update
                if not args.accumulate or (batch_id +
                                           1) % args.accumulate == 0:
                    trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(args.accumulate if args.accumulate else 1,
                                   ignore_stale_grad=True)
                    step_num += 1
                    if args.accumulate and args.accumulate > 1:
                        # set grad to zero for gradient accumulation
                        all_model_params.zero_grad()
                    if batch_id == 0 and epoch_id == 0:
                        toc = time.time()
                        logging.info(
                            'Time cost for the first forward-backward =%.2fs',
                            toc - tic)
                batch_loss = sum([ls.asscalar() for ls in batch_loss])
                step_loss += batch_loss
                if (batch_id + 1) % (args.log_interval) == 0:
                    log_train(batch_id, len(train_data), step_loss,
                              args.log_interval, epoch_id,
                              trainer.learning_rate)
                    step_loss = 0
                if step_num >= num_train_steps:
                    logging.info('Finish training step: %d', step_num)
                    finish_flag = True
                    break

            mx.nd.waitall()

        # inference on dev data
        for segment, dev_data in dev_data_list:
            metric_nm, metric_val = evaluate(dev_data, metric, segment)
            if best_metric is None or metric_val >= best_metric:
                best_metric = metric_val
                patience = args.early_stop
            else:
                if args.early_stop is not None:
                    patience -= 1
            metric_history.append((epoch_id, metric_nm, metric_val))

        if not args.only_inference:
            # save params
            ckpt_name = 'model_xlnet_{0}_{1}.params'.format(
                args.task_name, epoch_id)
            params_saved = os.path.join(output_dir, ckpt_name)
            nlp.utils.save_parameters(model, params_saved)
            logging.info('params saved in: %s', params_saved)
            toc = time.time()
            logging.info('Time cost=%.2fs', toc - tic)
            tic = toc

    if not args.only_inference:
        # we choose the best model based on metric[0],
        # assuming higher score stands for better model quality
        metric_history.sort(key=lambda x: x[2][0], reverse=True)
        epoch_id, metric_nm, metric_val = metric_history[0]
        ckpt_name = 'model_xlnet_{0}_{1}.params'.format(
            args.task_name, epoch_id)
        params_saved = os.path.join(output_dir, ckpt_name)
        nlp.utils.load_parameters(model, params_saved)
        metric_str = 'Best model at epoch {}. Validation metrics:'.format(
            epoch_id + 1)
        metric_str += ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(metric_str, *metric_val)

    # inference on test data
    for segment, test_data in test_data_list:
        test(test_data, segment)
    print('finish test!')


def evaluate(loader_dev, metric, segment):
    """Evaluate the model on validation dataset."""
    logging.info('Now we are doing evaluation on %s with %s.', segment, ctxs)
    metric.reset()
    step_loss = 0
    tic = time.time()
    out_list = []
    label_list = []
    for batch_id, seqs in enumerate(loader_dev):
        batch_loss = []
        # forward and backward
        data_list = list(split_and_load(seqs, ctxs))
        for splited_data in data_list:
            input_ids, valid_length, segment_ids, label = splited_data
            out = model(input_ids, segment_ids, valid_length=valid_length)
            batch_loss.append(loss_function(out, label).mean() / len(ctxs))
            if not do_regression:
                label = label.reshape((-1))
            out_list.append(out.as_in_context(mx.cpu(0)))
            label_list.append(label.as_in_context(mx.cpu(0)))

        batch_loss = sum([ls.asscalar() for ls in batch_loss])
        step_loss += batch_loss
        if (batch_id + 1) % (args.log_interval) == 0:
            log_eval(batch_id, len(loader_dev), step_loss, args.log_interval)
            step_loss = 0

    label_list = mx.nd.concat(*label_list, dim=0)
    out_list = mx.nd.concat(*out_list, dim=0)
    metric.update([label_list], [out_list])
    metric_nm, metric_val = log_metric(metric, is_training=False)
    mx.nd.waitall()
    toc = time.time()
    logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                 args.dev_batch_size * len(loader_dev) / (toc - tic))
    return metric_nm, metric_val


if __name__ == '__main__':
    train(task.metrics)
