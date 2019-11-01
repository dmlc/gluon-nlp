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
import multiprocessing
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from XLNet_classifier import XLNetClassifier
from transformer import model

sys.path.append('../bert/data')
#pylint: disable=wrong-import-position
from classification import MRPCTask, QQPTask, RTETask, STSBTask, SSTTask, \
     QNLITask, CoLATask, MNLITask, WNLITask, XNLITask, LCQMCTask, ChnSentiCorpTask
from data.transform import XLNetDatasetTransform





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

parser.add_argument(
    '--epochs', type=int, default=3, help='number of epochs.')

parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size. Number of examples per gpu in a minibatch.')

parser.add_argument(
    '--dev_batch_size',
    type=int,
    default=8,
    help='Batch size for dev set and test set')

parser.add_argument(
    '--lr',
    type=float,
    default=5e-5,
    help='Initial learning rate')

parser.add_argument(
    '--epsilon',
    type=float,
    default=1e-6,
    help='Small value to avoid division by 0'
)
parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0.1,
    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument(
    '--log_interval',
    type=int,
    default=10,
    help='report interval')
parser.add_argument(
    '--max_len',
    type=int,
    default=128,
    help='Maximum length of the sentence pairs')

parser.add_argument(
    '--pad',
    default=True,
    action='store_true',
    help='Whether to pad to maximum length when preparing data batches. '
         'Have to be true currently due to left padding')

parser.add_argument(
    '--seed', type=int, default=2, help='Random seed')

parser.add_argument(
    '--accumulate',
    type=int,
    default=None,
    help='The number of batches for gradients accumulation to simulate large batch size. '
         'Default is None')
parser.add_argument(
    '--gpu', type=int, default=None, help='Number of gpus for finetuning.')
parser.add_argument(
    '--cpu', type=int, default=None, help='Number of cpus for finetuning.')
parser.add_argument(
    '--task_name',
    default='MRPC',
    type=str,
    help='The name of the task to fine-tune.')

parser.add_argument(
    '--model',
    type=str,
    default='xlnet_cased_l12_h768_a12',
    help='The name of pre-trained XLNet model to fine-tune')

parser.add_argument(
    '--dataset',
    type=str,
    default='126gb',
    help='The dataset BERT pre-trained with.')

parser.add_argument(
    '--output_dir',
    type=str,
    default='./output_dir',
    help='The output directory where the model params will be written.')

parser.add_argument(
    '--only_inference',
    action='store_true',
    help='If set, we skip training and only perform inference on dev and test data.')

parser.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float16'],
    help='The data type for training. Doesnt support float16 currently')

parser.add_argument(
    '--model_parameters',
    type=str,
    default=None,
    help='A parameter file for the model that is loaded into the model'
    ' before training/inference. It is different from the parameter'
    ' file written after the model is trained.')

parser.add_argument(
    '--early_stop',
    type=int,
    default=None,
    help='Whether to perform early stopping based on the metric on dev set. '
         'The provided value is the patience. ')


args = parser.parse_args()



def split_and_load(arrs, ctx):
    """split and load arrays to a list of contexts"""
    assert isinstance(arrs, (list, tuple))
    # split and load
    loaded_arrs = [mx.gluon.utils.split_and_load(arr, ctx, even_split=False) for arr in arrs]
    return zip(*loaded_arrs)


logging.getLogger().setLevel(logging.INFO)
logging.captureWarnings(True)
logging.info(args)

batch_size = args.batch_size
dev_batch_size = args.dev_batch_size
task_name = args.task_name
lr = args.lr
epsilon = args.epsilon
accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval

if accumulate:
    logging.info('Using gradient accumulation. Effective batch size = ' \
                 'batch_size * accumulate = %d', accumulate * batch_size)

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)


num_workers = 0
ctxs = [mx.cpu(0)] if not args.gpu else [mx.gpu(i) for i in range(args.gpu)]

task = tasks[task_name]

# data type with mixed precision training
if args.dtype == 'float16':
    try:
        from mxnet.contrib import amp # pylint: disable=ungrouped-imports
        # monkey patch amp list since topk does not support fp16
        amp.lists.symbol.FP32_FUNCS.append('topk')
        amp.lists.symbol.FP16_FP32_FUNCS.remove('topk')
        amp.init()
    except ValueError:
        # topk is already in the FP32_FUNCS list
        amp.init()
    except ImportError:
        # amp is not available
        logging.info('Mixed precision training with float16 requires MXNet >= '
                     '1.5.0b20190627. Please consider upgrading your MXNet version.')
        sys.exit()

# model and loss
only_inference = args.only_inference
model_name = args.model
dataset = args.dataset

model_parameters = args.model_parameters
if only_inference and not model_parameters:
    warnings.warn('model_parameters is not set. '
                  'Randomly initialized model will be used for inference.')

get_pretrained = True

get_model_params = {
    'name' : model_name,
    'dataset_name' : dataset,
    'pretrained' : get_pretrained,
    'ctx' : ctxs,
    'use_decoder' : False,
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
model = XLNetClassifier(xlnet_base, dropout=0.1, num_classes=num_classes)


num_ctxes = len(ctxs)



# initialize classifier
if not model_parameters:
    model.classifier.initialize(init=initializer, ctx=ctxs)
    model.pooler.initialize(init=initializer, ctx=ctxs)

# load checkpointing
output_dir = args.output_dir

if model_parameters:
    logging.info('loading model params from %s', model_parameters)
    nlp.utils.load_parameters(model, model_parameters, ctx=ctxs, cast_dtype=True)

nlp.utils.mkdir(output_dir)

logging.debug(model)
model.hybridize(static_alloc=True)
loss_function.hybridize(static_alloc=True)

# data processing
do_lower_case = 'uncased' in dataset


def preprocess_data(_tokenizer, _task, _batch_size, _dev_batch_size, max_len, _vocab, pad=False):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()

    # transformation for data train and dev
    label_dtype = 'float32' if not _task.class_labels else 'int32'
    trans = XLNetDatasetTransform(_tokenizer, max_len,
                                  vocab=_vocab,
                                  class_labels=_task.class_labels,
                                  label_alias=_task.label_alias,
                                  pad=pad, pair=_task.is_pair,
                                  has_label=True)

    # data train
    # _task.dataset_train returns (segment_name, dataset)
    train_tsv = _task.dataset_train()[1]
    data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_tsv))
    data_train_len = data_train.transform(
        lambda input_id, length, segment_id, label_id: length, lazy=False)
    # bucket sampler for training
    pad_val = _vocab[_vocab.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val), # input
        nlp.data.batchify.Stack(),                      # length
        nlp.data.batchify.Pad(axis=0, pad_val=0),       # segment
        nlp.data.batchify.Stack(label_dtype))           # label

    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        data_train_len,
        batch_size=_batch_size,
        num_buckets=10,
        ratio=0,
        shuffle=True)
    # data loader for training
    loader_train = gluon.data.DataLoader(
        dataset=data_train,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)

    # data dev. For MNLI, more than one dev set is available
    dev_tsv = _task.dataset_dev()
    dev_tsv_list = dev_tsv if isinstance(dev_tsv, list) else [dev_tsv]
    loader_dev_list = []
    for segment, data in dev_tsv_list:
        data_dev = mx.gluon.data.SimpleDataset(pool.map(trans, data))
        loader_dev = mx.gluon.data.DataLoader(
            data_dev,
            batch_size=_dev_batch_size,
            num_workers=num_workers,
            shuffle=False,
            batchify_fn=batchify_fn)
        loader_dev_list.append((segment, loader_dev))

    # batchify for data test
    test_batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=0))
    # transform for data test
    test_trans = XLNetDatasetTransform(_tokenizer, max_len,
                                       vocab=_vocab,
                                       class_labels=None,
                                       pad=pad, pair=_task.is_pair,
                                       has_label=False)

    # data test. For MNLI, more than one test set is available
    test_tsv = _task.dataset_test()
    test_tsv_list = test_tsv if isinstance(test_tsv, list) else [test_tsv]
    loader_test_list = []
    for segment, data in test_tsv_list:
        data_test = mx.gluon.data.SimpleDataset(pool.map(test_trans, data))
        loader_test = mx.gluon.data.DataLoader(
            data_test,
            batch_size=_dev_batch_size,
            num_workers=num_workers,
            shuffle=False,
            batchify_fn=test_batchify_fn)
        loader_test_list.append((segment, loader_test))
    pool.close()
    return loader_train, loader_dev_list, loader_test_list, len(data_train)


# Get the loader.
logging.info('processing dataset...')
train_data, dev_data_list, test_data_list, num_train_examples = preprocess_data(
    tokenizer, task, batch_size, dev_batch_size, args.max_len, vocab, args.pad)


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
                 dev_batch_size * len(loader_test) / (toc - tic))
    # write result to a file.
    segment = segment.replace('_mismatched', '-mm')
    segment = segment.replace('_matched', '-m')
    segment = segment.replace('SST', 'SST-2')
    filename = args.task_name + segment.replace('test', '') + '.tsv'
    test_path = os.path.join(args.output_dir, filename)
    with io.open(test_path, 'w', encoding='utf-8') as f:
        f.write(u'index\tprediction\n')
        for i, pred in enumerate(results):
            f.write(u'%d\t%s\n'%(i, str(pred)))


def log_train(batch_id, batch_num, metric, step_loss, _log_interval, epoch_id, learning_rate):
    """Generate and print out the log message for training. """
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]

    train_str = '[Epoch %d Batch %d/%d] loss=%.4f, lr=%.7f, metrics:' + \
                ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(train_str, epoch_id + 1, batch_id + 1, batch_num,
                 step_loss / _log_interval, learning_rate, *metric_val)


def log_eval(batch_id, batch_num, metric, step_loss, _log_interval):
    """Generate and print out the log message for inference. """
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]

    eval_str = '[Batch %d/%d] loss=%.4f, metrics:' + \
               ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(eval_str, batch_id + 1, batch_num,
                 step_loss / _log_interval, *metric_val)


def train(metric):
    """Training function."""
    if not only_inference:
        logging.info('Now we are doing XLNet classification training on %s!', ctxs)

    all_model_params = model.collect_params()
    optimizer_params = {'learning_rate': lr, 'epsilon': epsilon, 'wd': 0.01}
    trainer = gluon.Trainer(all_model_params, 'adam',
                            optimizer_params, update_on_kvstore=False)
    if args.dtype == 'float16':
        amp.init_trainer(trainer)

    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in all_model_params.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if accumulate and accumulate > 1:
        for p in params:
            p.grad_req = 'add'
    # track best eval score
    metric_history = []
    best_metric = None
    patience = args.early_stop

    tic = time.time()

    for epoch_id in range(args.epochs):
        if args.early_stop and patience == 0:
            logging.info('Early stopping at epoch %d', epoch_id)
            break
        if not only_inference:
            metric.reset()
            step_loss = 0
            tic = time.time()
            all_model_params.zero_grad()
            for batch_id, seqs in enumerate(train_data):
                # learning rate schedule
                if step_num < num_warmup_steps:
                    new_lr = lr * step_num / num_warmup_steps
                else:
                    non_warmup_steps = step_num - num_warmup_steps
                    offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                    new_lr = lr - offset * lr
                trainer.set_learning_rate(new_lr)
                batch_loss = []
                out_list = []
                label_list = []
                # forward and backward
                with mx.autograd.record():
                    data_list = list(split_and_load(seqs, ctxs))
                    for splited_data in data_list:
                        input_ids, valid_length, segment_ids, label = splited_data
                        out = model(input_ids, segment_ids, valid_length=valid_length)
                        out_list.append(out)
                        label_list.append(label)
                        batch_loss.append(loss_function(out, label).mean())
                # update
                if not accumulate or (batch_id + 1) % accumulate == 0:
                    trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(accumulate if accumulate else 1, ignore_stale_grad=True)
                    step_num += 1
                    if accumulate and accumulate > 1:
                        # set grad to zero for gradient accumulation
                        all_model_params.zero_grad()
                batch_loss = sum([ls.asscalar() for ls in batch_loss])
                step_loss += batch_loss
                metric.update(label_list, out_list)
                if (batch_id + 1) % (args.log_interval) == 0:
                    log_train(batch_id, len(train_data), metric, step_loss, args.log_interval,
                              epoch_id, trainer.learning_rate)
                    step_loss = 0
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

        if not only_inference:
            # save params
            ckpt_name = 'model_xlnet_{0}_{1}.params'.format(task_name, epoch_id)
            params_saved = os.path.join(output_dir, ckpt_name)
            nlp.utils.save_parameters(model, params_saved)
            logging.info('params saved in: %s', params_saved)
            toc = time.time()
            logging.info('Time cost=%.2fs', toc - tic)
            tic = toc

    if not only_inference:
        # we choose the best model based on metric[0],
        # assuming higher score stands for better model quality
        metric_history.sort(key=lambda x: x[2][0], reverse=True)
        epoch_id, metric_nm, metric_val = metric_history[0]
        ckpt_name = 'model_xlnet_{0}_{1}.params'.format(task_name, epoch_id)
        params_saved = os.path.join(output_dir, ckpt_name)
        nlp.utils.load_parameters(model, params_saved)
        metric_str = 'Best model at epoch {}. Validation metrics:'.format(epoch_id + 1)
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
    for batch_id, seqs in enumerate(loader_dev):
        batch_loss = []
        out_list = []
        label_list = []
        # forward and backward
        batch_loss = []
        out_list = []
        label_list = []
        # forward and backward
        data_list = list(split_and_load(seqs, ctxs))
        for splited_data in data_list:
            input_ids, valid_length, segment_ids, label = splited_data
            out = model(input_ids, segment_ids, valid_length=valid_length)
            out_list.append(out)
            label_list.append(label)
            batch_loss.append(loss_function(out, label).mean())
            #batch_loss.append(loss_function(out, label).means())

        batch_loss = sum([ls.asscalar() for ls in batch_loss])
        step_loss += batch_loss
        metric.update(label_list, out_list)

        if (batch_id + 1) % (args.log_interval) == 0:
            log_eval(batch_id, len(loader_dev), metric, step_loss, args.log_interval)
            step_loss = 0

    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]
    metric_str = 'validation metrics:' + ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(metric_str, *metric_val)

    mx.nd.waitall()
    toc = time.time()
    logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                 dev_batch_size * len(loader_dev) / (toc - tic))
    return metric_nm, metric_val

if __name__ == '__main__':
    train(task.metrics)

sys.exit()
