"""
Question Answering with XLNet
"""
import os
import time
import argparse
import random
import logging
import warnings
import copy
import json
import collections
import pickle
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.data import SQuAD
from model.qa import XLNetForQA
from data.new_qa import SQuADTransform, preprocess_dataset, convert_examples_to_inputs
from transformer import model
from xlnet_qa_evaluate import predict_extended
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

os.environ['MXNET_USE_FUSION'] = '0'
log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s',
                              datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(description='XLNet QA example.'
                                 'We fine-tune the XLNet model on SQuAD dataset.')

parser.add_argument('--only_predict', action='store_true', help='Whether to predict only.')

parser.add_argument('--model_parameters', type=str, default=None, help='Model parameter file')

parser.add_argument('--model', type=str, default='xlnet_cased_l12_h768_a12',
                    help='The name of pre-trained XLNet model to fine-tune')

parser.add_argument('--dataset', type=str, default='126gb',
                    help='The dataset BERT pre-trained with.')

parser.add_argument('--predict_file', default='./data/dev-v2.0.json', type=str,
                    help='SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json')

parser.add_argument('--uncased', action='store_true',
                    help='if set, inputs are converted to lower case.')

parser.add_argument(
    '--output_dir', type=str, default='./output_dir',
    help='The output directory where the model params will be written.'
    ' default is ./output_dir')

parser.add_argument('--epochs', type=int, default=3, help='number of epochs, default is 3')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size', type=int, default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer', type=str, default='bertadam',
                    help='optimization algorithm. default is bertadam')

parser.add_argument(
    '--accumulate', type=int, default=None, help='The number of batches for '
    'gradients accumulation to simulate large batch size. Default is None')

parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate. default is 5e-5')

parser.add_argument(
    '--warmup_ratio', type=float, default=0,
    help='ratio of warmup steps that linearly increase learning rate from '
    '0 to target learning rate. default is 0')

parser.add_argument('--log_interval', type=int, default=10, help='report interval. default is 10')

parser.add_argument(
    '--max_seq_length', type=int, default=512,
    help='The maximum total input sequence length after WordPiece tokenization.'
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. default is 512')

parser.add_argument(
    '--doc_stride', type=int, default=128,
    help='When splitting up a long document into chunks, how much stride to '
    'take between chunks. default is 128')

parser.add_argument(
    '--max_query_length', type=int, default=64,
    help='The maximum number of tokens for the question. Questions longer than '
    'this will be truncated to this length. default is 64')

parser.add_argument(
    '--n_best_size', type=int, default=20,
    help='The total number of n-best predictions to generate in the '
    'nbest_predictions.json output file. default is 20')

parser.add_argument(
    '--max_answer_length', type=int, default=64,
    help='The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another.'
    ' default is 64')

parser.add_argument('--version_2', action='store_true',
                    help='SQuAD examples whether contain some that do not have an answer.')

parser.add_argument(
    '--null_score_diff_threshold', type=float, default=0.0,
    help='If null_score - best_non_null is greater than the threshold predict null.'
    'Typical values are between -1.0 and -5.0. default is 0.0')

parser.add_argument('--gpu', type=int, default=None,
                    help='Number of gpus to use for finetuning. CPU is used if not set.')

parser.add_argument('--sentencepiece', type=str, default=None,
                    help='Path to the sentencepiece .model file for both tokenization and vocab.')

parser.add_argument('--debug', action='store_true',
                    help='Run the example in test mode for sanity checks')
parser.add_argument('--pretrained_xlnet_parameters', type=str, default=None,
                    help='Pre-trained bert model parameter file. default is None')

parser.add_argument('--layerwise_decay', type=float, default=0.75, help='Layer-wise lr decay')
parser.add_argument('--wd', type=float, default=0.01, help='adam weight decay')
parser.add_argument('--seed', type=int, default=29, help='Random seed')
parser.add_argument('--start_top_n', type=int, default=5, help='to be added')
parser.add_argument('--end_top_n', type=int, default=5, help='to be added')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--attention_dropout', type=float, default=0.1, help='attention dropout')
parser.add_argument('--training_steps', type=int, help='training steps')
parser.add_argument('--raw', action='store_true', help='if do data preprocessing or load from pickled file')
parser.add_argument('--dev_dataset_file', default='./output_dir/out.dev', type=str, help='location of dev dataset')
parser.add_argument('--train_dataset_file', default='./output_dir/out.train', type=str, help='location of train dataset')

args = parser.parse_args()

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

fh = logging.FileHandler(os.path.join(args.output_dir, 'finetune_squad.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

log.info(args)

pretrained_xlnet_parameters = args.pretrained_xlnet_parameters
if pretrained_xlnet_parameters and args.model_parameters:
    raise ValueError('Cannot provide both pre-trained BERT parameters and '
                     'BertForQA model parameters.')

ctx = [mx.cpu(0)] if not args.gpu else [mx.gpu(i) for i in range(args.gpu)]

log_interval = args.log_interval * args.accumulate if args.accumulate else args.log_interval
if args.accumulate:
    log.info('Using gradient accumulation. Effective batch size = %d',
             args.accumulate * args.batch_size)
if args.max_seq_length <= args.max_query_length + 3:
    raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                     '(%d) + 3' % (args.max_seq_length, args.max_query_length))

# vocabulary and tokenizer

get_pretrained = True

get_model_params = {
    'name': args.model,
    'dataset_name': args.dataset,
    'pretrained': get_pretrained,
    'ctx': ctx,
    'use_decoder': False,
    'dropout': args.dropout,
    'attention_dropout': args.attention_dropout
}

xlnet_base, vocab, tokenizer = model.get_model(**get_model_params)

num_layers = len(xlnet_base._net.transformer_cells)
for (i, layer_parameters) in enumerate(xlnet_base._net.transformer_cells):
    layer_params = layer_parameters.collect_params()
    for key, value in layer_params.items():
        value.lr_mult = args.layerwise_decay**(num_layers - i - 1)

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Stack(),  # Already padded in data transform
    nlp.data.batchify.Stack(),  # Already padded in data transform
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'))

if pretrained_xlnet_parameters:
    # only load XLnetModel parameters
    nlp.utils.load_parameters(xlnet_base, pretrained_xlnet_parameters, ctx=ctx, ignore_extra=True,
                              cast_dtype=True)

net = XLNetForQA(xlnet_base=xlnet_base, start_top_n=args.start_top_n, end_top_n=args.end_top_n,
                 version_2=args.version_2)
net_eval = XLNetForQA(xlnet_base=xlnet_base, start_top_n=args.start_top_n, end_top_n=args.end_top_n,
                      version_2=args.version_2, is_eval=True, params=net.collect_params())

initializer = mx.init.Normal(0.02)

if args.model_parameters:
    # load complete XLNetForQA parameters
    nlp.utils.load_parameters(net, args.model_parameters, ctx=ctx, cast_dtype=True)
else:
    net.start_logits.initialize(init=initializer, ctx=ctx)
    net.end_logits.initialize(init=initializer, ctx=ctx)
    if args.version_2:
        net.answer_class.initialize(init=initializer, ctx=ctx)

net.hybridize(static_alloc=True)
net_eval.hybridize(static_alloc=True)


def split_array(arr, num_of_splits):
    """split an array into a number of splits"""
    size = arr.shape[0]
    if size < num_of_splits:
        return [arr[i:i + 1] for i in range(size)]
    slice_len, rest = divmod(size, num_of_splits)
    div_points = [0] + [(slice_len * index + min(index, rest) + slice_len + (index < rest))
                        for index in range(num_of_splits)]
    slices = [arr[div_points[i]:div_points[i + 1]] for i in range(num_of_splits)]
    return slices


def split_and_load(arrs, ctxs):
    """split and load arrays to a list of contexts"""
    assert isinstance(arrs, (list, tuple))
    # split and load
    loaded_arrs = [[i.as_in_context(ctx) for i, ctx in zip(split_array(arr, len(ctxs)), ctxs)]
                   for arr in arrs]
    return zip(*loaded_arrs)


def train():
    """Training function."""
    segment = 'train' if not args.debug else 'dev'
    log.info('Loading %s data...', segment)
    if args.version_2:
        train_data = SQuAD(segment, version='2.0')
    else:
        train_data = SQuAD(segment, version='1.1')
    if args.debug:
        sampled_data = [train_data[i] for i in range(100)]
        train_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in Train data: %s', len(train_data))
    if args.raw:
        train_data_transform = preprocess_dataset(
            train_data,
            SQuADTransform(copy.copy(tokenizer), vocab, max_seq_length=args.max_seq_length,
                           doc_stride=args.doc_stride, max_query_length=args.max_query_length,
                           is_pad=True, is_training=True), dataset_file=args.train_dataset_file)
    else:
        train_data_transform = preprocess_dataset(raw=False, dataset_file=args.train_dataset_file)

    log.info('The number of examples after preprocessing: %s', len(train_data_transform))

    train_dataloader = mx.gluon.data.DataLoader(train_data_transform, batchify_fn=batchify_fn,
                                                batch_size=args.batch_size, num_workers=4,
                                                shuffle=True)

    optimizer_params = {'learning_rate': args.lr, 'wd': args.wd}
    try:
        trainer = mx.gluon.Trainer(net.collect_params(), args.optimizer, optimizer_params,
                                   update_on_kvstore=False)
    except ValueError as _:
        warnings.warn('AdamW optimizer is not found. Please consider upgrading to '
                      'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')
        trainer = mx.gluon.Trainer(net.collect_params(), 'bertadam', optimizer_params,
                                   update_on_kvstore=False)

    num_train_examples = len(train_data_transform)
    step_size = args.batch_size * args.accumulate if args.accumulate else args.batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    epoch_number = args.epochs
    if args.training_steps:
        num_train_steps = args.training_steps
        epoch_number = 999

    log.info('training steps=%d', num_train_steps)
    num_warmup_steps = int(num_train_steps * args.warmup_ratio)
    step_num = 0

    def set_new_lr(step_num, batch_id):
        """set new learning rate"""
        # set grad to zero for gradient accumulation
        if args.accumulate:
            if batch_id % args.accumulate == 0:
                net.collect_params().zero_grad()
                step_num += 1
        else:
            step_num += 1
        # learning rate schedule
        # Notice that this learning rate scheduler is adapted from traditional linear learning
        # rate scheduler where step_num >= num_warmup_steps, new_lr = 1 - step_num/num_train_steps
        if step_num < num_warmup_steps:
            new_lr = args.lr * step_num / num_warmup_steps
        else:
            offset = (step_num - num_warmup_steps) * args.lr / \
                (num_train_steps - num_warmup_steps)
            new_lr = args.lr - offset
        trainer.set_learning_rate(new_lr)
        return step_num

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']
    # Set grad_req if gradient accumulation is required
    if args.accumulate:
        for p in params:
            p.grad_req = 'add'

    epoch_tic = time.time()
    total_num = 0
    log_num = 0
    finish_flag = False
    for epoch_id in range(epoch_number):
        step_loss = 0.0
        step_loss_span = 0
        step_loss_cls = 0
        tic = time.time()
        if finish_flag:
            break
        for batch_id, data in enumerate(train_dataloader):
            # set new lr
            step_num = set_new_lr(step_num, batch_id)
            data_list = list(split_and_load(data, ctx))
            # forward and backward
            batch_loss = []
            batch_loss_sep = []
            with mx.autograd.record():
                for splited_data in data_list:
                    _, inputs, token_types, valid_length, p_mask, start_label, end_label, _is_impossible = splited_data  # pylint: disable=line-too-long
                    valid_length = valid_length.astype('float32')
                    is_impossible = _is_impossible if args.version_2 else None
                    log_num += len(inputs)
                    total_num += len(inputs)
                    out_sep, out = net(
                        inputs,
                        token_types,
                        valid_length,
                        [start_label, end_label],
                        p_mask=p_mask,  # pylint: disable=line-too-long
                        is_impossible=is_impossible)
                    ls = out.mean() / len(ctx)
                    if args.accumulate:
                        ls = ls / args.accumulate
                    batch_loss_sep.append(out_sep)
                    batch_loss.append(ls)
                    ls.backward()
            # update
            if not args.accumulate or (batch_id + 1) % args.accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1, ignore_stale_grad=True)
            if args.version_2:
                step_loss_sep_tmp = np.array([[span_ls.mean().asscalar(), cls_ls.mean().asscalar()] for span_ls, cls_ls in batch_loss_sep])
                step_loss_sep_tmp = list(np.sum(step_loss_sep_tmp, axis=0))
                step_loss_span += step_loss_sep_tmp[0]
                step_loss_cls += step_loss_sep_tmp[1]

            step_loss += sum([ls.asscalar() for ls in batch_loss])
            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                log.info(
                    'Epoch: %d, Batch: %d/%d, Loss=%.4f, lr=%.7f Time cost=%.1f Thoughput=%.2f samples/s'  # pylint: disable=line-too-long
                    ,
                    epoch_id + 1,
                    batch_id + 1,
                    len(train_dataloader),
                    step_loss / log_interval,
                    trainer.learning_rate,
                    toc - tic,
                    log_num / (toc - tic))

                if args.version_2:
                    if args.accumulate:
                        step_loss_span = step_loss_span / args.accumulate
                        step_loss_cls = step_loss_cls / args.accumulate
                    log.info('span_loss: %.4f, cls_loss: %.4f', step_loss_span / log_interval, step_loss_cls / log_interval)

                tic = time.time()
                step_loss = 0.0
                step_loss_span = 0
                step_loss_cls = 0
                log_num = 0
            if step_num >= num_train_steps:
                logging.info('Finish training step: %d', step_num)
                finish_flag = True
                break
        epoch_toc = time.time()
        log.info('Time cost=%.2f s, Thoughput=%.2f samples/s', epoch_toc - epoch_tic,
                 total_num / (epoch_toc - epoch_tic))
        ckpt_name = 'model_xlnet_squad_{0}.params'.format(epoch_id + 1)
        params_saved = os.path.join(args.output_dir, ckpt_name)
        nlp.utils.save_parameters(net, params_saved)
        log.info('params saved in: %s', params_saved)


RawResultExtended = collections.namedtuple(
    'RawResultExtended',
    ['start_top_log_probs', 'start_top_index', 'end_top_log_probs', 'end_top_index', 'cls_logits'])


def evaluate(prefix='p'):
    """Evaluate the model on validation dataset.
    """
    log.info('Loading dev data...')
    if args.version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    (_, _), (data_file_name, _) \
        = dev_data._data_file[dev_data._version][dev_data._segment]
    dev_data_path = os.path.join(dev_data._root, data_file_name)

    if args.debug:
        sampled_data = [dev_data[0], dev_data[1], dev_data[2]]
        dev_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in dev data: %d', len(dev_data))


    if args.raw:
        dev_dataset = dev_data.transform(
            SQuADTransform(copy.copy(tokenizer), vocab, max_seq_length=args.max_seq_length,
                           doc_stride=args.doc_stride, max_query_length=args.max_query_length,
                           is_pad=True, is_training=False)._transform, lazy=False)
        with open(args.dev_dataset_file, 'wb') as file:
            pickle.dump(list(dev_dataset), file)
    else:
        with open(args.dev_dataset_file , 'rb') as file:
            dev_dataset = pickle.load(file)
            dev_dataset = mx.gluon.data.SimpleDataset(dev_dataset)

    dev_data_transform = convert_examples_to_inputs(dev_dataset)

    log.info('The number of examples after preprocessing: %d', len(dev_data_transform))

    dev_dataloader = mx.gluon.data.DataLoader(dev_data_transform, batchify_fn=batchify_fn,
                                              num_workers=4, batch_size=args.test_batch_size,
                                              shuffle=False, last_batch='keep')

    log.info('start prediction')

    all_results = collections.defaultdict(list)

    epoch_tic = time.time()
    total_num = 0
    for (batch_id, data) in enumerate(dev_dataloader):
        data_list = list(split_and_load(data, ctx))
        for splited_data in data_list:
            example_ids, inputs, token_types, valid_length, p_mask, _, _, _ = splited_data
            total_num += len(inputs)
            outputs = net_eval(inputs, token_types, valid_length, p_mask=p_mask)
            example_ids = example_ids.asnumpy().tolist()
            for c, example_ids in enumerate(example_ids):
                result = RawResultExtended(
                    start_top_log_probs=outputs[0][c].asnumpy().tolist(),
                    start_top_index=outputs[1][c].asnumpy().tolist(),
                    end_top_log_probs=outputs[2][c].asnumpy().tolist(),
                    end_top_index=outputs[3][c].asnumpy().tolist(),
                    cls_logits=outputs[4][c].asnumpy().tolist()
                    if outputs[4] is not None else [-1e30])
                all_results[example_ids].append(result)
        if batch_id % args.log_interval == 0:
            log.info('Batch: %d/%d', batch_id + 1, len(dev_dataloader))

    epoch_toc = time.time()
    log.info('Time cost=%2f s, Thoughput=%.2f samples/s', epoch_toc - epoch_tic,
             total_num / (epoch_toc - epoch_tic))

    log.info('Get prediction results...')

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for features in dev_dataset:
        results = all_results[features[0].example_id]
        example_qas_id = features[0].qas_id
        score_diff, best_non_null_entry, nbest_json = predict_extended(
            features=features, results=results,
            sp_model=nlp.data.SentencepieceTokenizer(tokenizer._sentencepiece_path)._processor,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length, start_n_top=args.start_top_n,
            end_n_top=args.end_top_n)
        scores_diff_json[example_qas_id] = score_diff
        all_predictions[example_qas_id] = best_non_null_entry
        all_nbest_json[example_qas_id] = nbest_json

    output_prediction_file = os.path.join(args.output_dir, 'predictions_{}.json'.format(prefix))
    output_nbest_file = os.path.join(args.output_dir, 'nbest_predictions_{}.json'.format(prefix))
    if args.version_2:
        output_null_log_odds_file = os.path.join(args.output_dir,
                                                 'null_odds_{}.json'.format(prefix))
    else:
        output_null_log_odds_file = None

    with open(output_prediction_file, 'w') as writer:
        writer.write(json.dumps(all_predictions, indent=4) + '\n')
    with open(output_nbest_file, 'w') as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
    if args.version_2:
        with open(output_null_log_odds_file, 'w') as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + '\n')

    if args.version_2:
        evaluate_options = EVAL_OPTS(data_file=dev_data_path, pred_file=output_prediction_file,
                                     na_prob_file=output_null_log_odds_file,
                                     na_prob_thresh=args.null_score_diff_threshold)
    else:
        evaluate_options = EVAL_OPTS(data_file=dev_data_path, pred_file=output_prediction_file,
                                     na_prob_file=None, na_prob_thresh=args.null_score_diff_threshold)

    results = evaluate_on_squad(evaluate_options)
    return results


if __name__ == '__main__':
    if not args.only_predict:
        train()
        evaluate()
    else:
        evaluate()
