"""
Question Answering with XLNet
"""
# pylint:disable=redefined-outer-name,logging-format-interpolation

import os
import time
import argparse
import random
import logging
import warnings
import json
import collections
import pickle
import sys
import itertools
import subprocess
import multiprocessing as mp
from functools import partial
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.data import SQuAD
from gluonnlp.data.bert.glue import concat_sequences
from gluonnlp.data.bert.squad import get_doc_spans, \
    check_is_max_context, convert_squad_examples, align_position2doc_spans
from gluonnlp.data.xlnet.squad import lcs_match, convert_index
from model.qa import XLNetForQA
from transformer import model
from xlnet_qa_evaluate import predict_extended
parser = argparse.ArgumentParser(description='XLNet QA example.'
                                 'We fine-tune the XLNet model on SQuAD dataset.')

# I/O configuration
parser.add_argument('--sentencepiece', type=str, default=None,
                    help='Path to the sentencepiece .model file for both tokenization and vocab.')
parser.add_argument('--pretrained_xlnet_parameters', type=str, default=None,
                    help='Pre-trained bert model parameter file. default is None')
parser.add_argument('--load_pickle', action='store_true',
                    help='Whether do data preprocessing or load from pickled file')
parser.add_argument('--dev_dataset_file', default='./output_dir/out.dev', type=str,
                    help='Path to dev data features')
parser.add_argument('--train_dataset_file', default='./output_dir/out.train', type=str,
                    help='Path to train data features')
parser.add_argument('--model_parameters', type=str, default=None, help='Model parameter file')
parser.add_argument(
    '--output_dir', type=str, default='./output_dir',
    help='The output directory where the model params will be written.'
    ' default is ./output_dir')

# Training configuration
parser.add_argument('--seed', type=int, default=3, help='Random seed')
parser.add_argument('--version_2', action='store_true', help='Whether use SQuAD v2.0 dataset')
parser.add_argument('--model', type=str, default='xlnet_cased_l12_h768_a12',
                    choices=['xlnet_cased_l24_h1024_a16', 'xlnet_cased_l12_h768_a12'],
                    help='The name of pre-trained XLNet model to fine-tune')
parser.add_argument('--dataset', type=str, default='126gb', choices=['126gb'],
                    help='The dataset BERT pre-trained with. Currently only 126gb is available')
parser.add_argument(
    '--uncased', action='store_true', help=
    'if set, inputs are converted to lower case. Up to 01/04/2020, all released models are cased')
parser.add_argument('--gpu', type=int, default=None,
                    help='Number of gpus to use for finetuning. CPU is used if not set.')
parser.add_argument('--log_interval', type=int, default=10, help='report interval. default is 10')
parser.add_argument('--debug', action='store_true',
                    help='Run the example in test mode for sanity checks')
parser.add_argument('--only_predict', action='store_true', help='Whether to predict only.')

# Hyperparameters
parser.add_argument('--epochs', type=int, default=3, help='number of epochs, default is 3')
parser.add_argument(
    '--training_steps', type=int, help='training steps. Note that epochs will be ignored '
    'if training steps are set')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size', type=int, default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer', type=str, default='bertadam',
                    help='optimization algorithm. default is bertadam')

parser.add_argument(
    '--accumulate', type=int, default=None, help='The number of batches for '
    'gradients accumulation to simulate large batch size. Default is None')

parser.add_argument('--lr', type=float, default=3e-5,
                    help='Initial learning rate. default is 5e-5')

parser.add_argument(
    '--warmup_ratio', type=float, default=0,
    help='ratio of warmup steps that linearly increase learning rate from '
    '0 to target learning rate. default is 0')
parser.add_argument('--layerwise_decay', type=float, default=0.75, help='Layer-wise lr decay')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--attention_dropout', type=float, default=0.1, help='attention dropout')

# Data pre/post processing
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
    '--round_to', type=int, default=None,
    help='The length of padded sequences will be rounded up to be multiple of this argument.'
         'When round to is set to 8, training throughput may increase for mixed precision'
         'training on GPUs with tensorcores.')

parser.add_argument('--start_top_n', type=int, default=5,
                    help='Number of start-position candidates')
parser.add_argument('--end_top_n', type=int, default=5,
                    help='Number of end-position candidates corresponding '
                    'to a start position')
parser.add_argument('--n_best_size', type=int, default=5, help='top N results written to file')
parser.add_argument(
    '--max_answer_length', type=int, default=64,
    help='The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another.'
    ' default is 64')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers used for data preprocessing')
parser.add_argument(
    '--null_score_diff_threshold', type=float, default=0.0,
    help='If null_score - best_non_null is greater than the threshold predict null.'
    'Typical values are between -1.0 and -5.0. default is 0.0. '
    'Note that a best value can be automatically found by the evaluation script')

args = parser.parse_args()

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# set the logger
log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s',
                              datefmt='%H:%M:%S')
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

# model, vocabulary and tokenizer
xlnet_base, vocab, tokenizer = model.get_model(**get_model_params)

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack('int32'),  # example_id
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], dtype='int32',
                          round_to=args.round_to),  # input_ids
    nlp.data.batchify.Pad(axis=0, pad_val=3, dtype='int32', round_to=args.round_to),  # segment_ids
    nlp.data.batchify.Stack('float32'),  # valid_length
    nlp.data.batchify.Pad(axis=0, pad_val=1, round_to=args.round_to),  # p_mask
    nlp.data.batchify.Stack('float32'),  # start_position
    nlp.data.batchify.Stack('float32'),  # end_position
    nlp.data.batchify.Stack('float32'))  # is_impossible

if pretrained_xlnet_parameters:
    # only load XLnetModel parameters
    nlp.utils.load_parameters(xlnet_base, pretrained_xlnet_parameters, ctx=ctx, ignore_extra=True,
                              cast_dtype=True)

units = xlnet_base._net._units
net = XLNetForQA(xlnet_base=xlnet_base, start_top_n=args.start_top_n, end_top_n=args.end_top_n,
                 units=units)

net_eval = XLNetForQA(xlnet_base=xlnet_base, start_top_n=args.start_top_n,
                      end_top_n=args.end_top_n, units=units, is_eval=True,
                      params=net.collect_params())

initializer = mx.init.Normal(0.02)

if args.model_parameters:
    # load complete XLNetForQA parameters
    nlp.utils.load_parameters(net, args.model_parameters, ctx=ctx, cast_dtype=True)
else:
    net.start_logits.initialize(init=initializer, ctx=ctx)
    net.end_logits.initialize(init=initializer, ctx=ctx)
    net.answer_class.initialize(init=initializer, ctx=ctx)

net.hybridize(static_alloc=True)
net_eval.hybridize(static_alloc=True)

SquadXLNetFeautre = collections.namedtuple('SquadXLNetFeautre', [
    'example_id', 'qas_id', 'valid_length', 'tokens', 'tok_start_to_orig_index',
    'tok_end_to_orig_index', 'token_is_max_context', 'input_ids', 'p_mask', 'segment_ids',
    'start_position', 'end_position', 'paragraph_text', 'paragraph_len', 'is_impossible'
])


def convert_examples_to_features(example, tokenizer=None, cls_token=None, sep_token=None,
                                 vocab=None, max_seq_length=384, doc_stride=128,
                                 max_query_length=64, is_training=True):
    """convert the examples to the XLNet features"""
    query_tokenized = tokenizer(example.question_text)[:max_query_length]
    #tokenize paragraph and get start/end position of the answer in tokenized paragraph
    paragraph_tokenized = tokenizer(example.paragraph_text)

    chartok_to_tok_index = [] # char to its corresponding token's index
    tok_start_to_chartok_index = [] # token index to its first character's index
    tok_end_to_chartok_index = [] # token index to its last character's index
    char_cnt = 0
    for i, token in enumerate(paragraph_tokenized):
        chartok_to_tok_index.extend([i] * len(token))
        tok_start_to_chartok_index.append(char_cnt)
        char_cnt += len(token)
        tok_end_to_chartok_index.append(char_cnt - 1)

    tok_cat_text = ''.join(paragraph_tokenized).replace(u'▁', ' ')

    # XLNet takes a more complicated strategy to match the origin text
    # and the tokenized tokens
    # Get the LCS matching between origin text and token-concatenated text.
    n, m = len(example.paragraph_text), len(tok_cat_text)
    max_dist = abs(n - m) + 5
    for _ in range(2):
        f, g = lcs_match(max_dist, example.paragraph_text, tok_cat_text)
        if f[n - 1, m - 1] > 0.8 * n:
            break
        max_dist *= 2

    # Get the mapping from orgin text/tokenized text to tokenized text/origin text
    orig_to_chartok_index = [None] * n
    chartok_to_orig_index = [None] * m
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        if (i, j) not in g:
            break
        if g[(i, j)] == 2:
            orig_to_chartok_index[i] = j
            chartok_to_orig_index[j] = i
            i, j = i - 1, j - 1
        elif g[(i, j)] == 1:
            j = j - 1
        else:
            i = i - 1

    # get start/end mapping
    tok_start_to_orig_index = []
    tok_end_to_orig_index = []
    for i in range(len(paragraph_tokenized)): # for each token in the tokenized paragraph
        start_chartok_pos = tok_start_to_chartok_index[i] # first character's index in origin text
        end_chartok_pos = tok_end_to_chartok_index[i] # last character's index in origin text
        start_orig_pos = convert_index(chartok_to_orig_index, start_chartok_pos, n, is_start=True)
        end_orig_pos = convert_index(chartok_to_orig_index, end_chartok_pos, m, is_start=False)

        tok_start_to_orig_index.append(start_orig_pos)
        tok_end_to_orig_index.append(end_orig_pos)

    tok_start_position, tok_end_position = -1, -1

    # get mapped start/end position
    if is_training and not example.is_impossible:
        start_chartok_pos = convert_index(orig_to_chartok_index, example.start_offset,
                                          is_start=True)
        tok_start_position = chartok_to_tok_index[start_chartok_pos]

        end_chartok_pos = convert_index(orig_to_chartok_index, example.end_offset, is_start=False)
        tok_end_position = chartok_to_tok_index[end_chartok_pos]
        assert tok_start_position <= tok_end_position

    # get doc spans using sliding window
    doc_spans, doc_spans_indices = get_doc_spans(paragraph_tokenized,
                                                 max_seq_length - len(query_tokenized) - 3,
                                                 doc_stride)

    # record whether the tokens in a docspan have max context
    token_is_max_context = [{
        p: check_is_max_context(doc_spans_indices, i, p + doc_spans_indices[i][0])
        for p in range(len(doc_span))
    } for (i, doc_span) in enumerate(doc_spans)]

    # get token -> origin text mapping
    cur_tok_start_to_orig_index = [[tok_start_to_orig_index[p + st] for p in range(len(doc_span))]
                                   for doc_span, (st, ed) in zip(doc_spans, doc_spans_indices)]
    cur_tok_end_to_orig_index = [[tok_end_to_orig_index[p + st] for p in range(len(doc_span))]
                                 for doc_span, (st, ed) in zip(doc_spans, doc_spans_indices)]

    # get sequence features: tokens, segment_ids, p_masks
    seq_features = [
        concat_sequences([doc_span, query_tokenized], [[sep_token]] * 2 + [[cls_token]],
                         [[0] * len(doc_span), [1] * len(query_tokenized)], [[1], [1], [0]])
        for doc_span in doc_spans
    ]

    # get the start/end positions aligned to doc spans. If is_impossible or position out of span
    # set position to cls_index, i.e., last token in the sequence.
    if not example.is_impossible:
        positions = [
            align_position2doc_spans([tok_start_position, tok_end_position], doc_idx, offset=0,
                                     default_value=len(seq[0]) - 1)
            for (doc_idx, seq) in zip(doc_spans_indices, seq_features)
        ]
    else:
        positions = [(len(seq_feature[0]) - 1, len(seq_feature[0]) - 1)
                     for seq_feature in seq_features]

    features = [
        SquadXLNetFeautre(example_id=example.example_id, qas_id=example.qas_id,
                          tok_start_to_orig_index=t2st, tok_end_to_orig_index=t2ed,
                          valid_length=len(tokens), tokens=tokens, token_is_max_context=is_max,
                          input_ids=vocab[tokens], p_mask=p_mask, segment_ids=segment_ids,
                          start_position=start, end_position=end,
                          paragraph_text=example.paragraph_text, paragraph_len=len(tokens),
                          is_impossible=(start == len(tokens) - 1))
        for (tokens, segment_ids, p_mask), (
            start,
            end), is_max, t2st, t2ed in zip(seq_features, positions, token_is_max_context,
                                            cur_tok_start_to_orig_index, cur_tok_end_to_orig_index)
    ]
    return features


def preprocess_dataset(tokenizer, dataset, vocab=None, max_seq_length=384, doc_stride=128,
                       max_query_length=64, num_workers=16, load_from_pickle=False,
                       feature_file=None, is_training=True):
    """Loads a dataset into features"""
    vocab = tokenizer.vocab if vocab is None else vocab
    trans = partial(convert_examples_to_features, tokenizer=tokenizer, cls_token=vocab.cls_token,
                    sep_token=vocab.sep_token, vocab=vocab, max_seq_length=max_seq_length,
                    doc_stride=doc_stride, max_query_length=max_query_length)
    pool = mp.Pool(num_workers)
    start = time.time()
    if not load_from_pickle:
        example_trans = partial(convert_squad_examples, is_training=is_training)
        # convert the raw dataset into raw features
        examples = pool.map(example_trans, dataset)
        raw_features = list(map(trans, examples))  #pool.map(trans, examples)
        if feature_file:
            with open(feature_file, 'wb') as file:
                pickle.dump(raw_features, file)
    else:
        assert feature_file, 'feature file should be provided.'
        with open(feature_file, 'rb') as file:
            raw_features = pickle.load(file)

    end = time.time()
    pool.close()
    log.info('Done! Transform dataset costs %.2f seconds.', (end - start))
    return raw_features


def convert_full_features_to_input_features(raw_features):
    """convert the full features into the input features"""
    data_features = mx.gluon.data.SimpleDataset(list(itertools.chain.from_iterable(raw_features)))
    data_features = data_features.transform(lambda *example: (
        example[0],  # example_id
        example[7],  # inputs_id
        example[9],  # segment_ids
        example[2],  # valid_length,
        example[8],  # p_mask
        example[10],  # start_position,
        example[11],  # end_position
        example[14]))  # is_impossible
    return data_features


def split_array(arr, num_of_splits):
    """split an array into equal pieces"""
    # TODO Replace this function with gluon.utils.split_data() once targeting MXNet 1.7
    size = arr.shape[0]
    if size < num_of_splits:
        return [arr[i:i + 1] for i in range(size)]
    slice_len, rest = divmod(size, num_of_splits)
    div_points = [0] + [(slice_len * index + min(index, rest) + slice_len + (index < rest))
                        for index in range(num_of_splits)]
    slices = [arr[div_points[i]:div_points[i + 1]] for i in range(num_of_splits)]
    return slices


def split_and_load(arrs, _ctxs):
    """split and load arrays to a list of contexts"""
    # TODO Replace split_array() with gluon.utils.split_data() once targeting MXNet 1.7
    assert isinstance(arrs, (list, tuple))
    # split and load
    loaded_arrs = [[i.as_in_context(ctx) for i, ctx in zip(split_array(arr, len(_ctxs)), _ctxs)]
                   for arr in arrs]
    return zip(*loaded_arrs)


def _apply_gradient_decay():
    """apply layer-wise gradient decay.

    Note that the description in origin paper about layer-wise learning rate decay
    is inaccurate. According to their implementation, they are actually performing
    layer-wise gradient decay. Gradient decay and learning rate decay could be the
    same by using standard SGD, but different by using Adaptive optimizer(e.g., Adam).
    """
    parameter_not_included = ['seg_emb', 'query_key_bias', 'query_emb_bias', 'query_seg_bias']
    num_layers = len(xlnet_base._net.transformer_cells)
    for (i, layer_parameters) in enumerate(xlnet_base._net.transformer_cells):
        layer_params = layer_parameters.collect_params()
        for key, value in layer_params.items():
            skip = False
            for pn in parameter_not_included:
                if pn in key:
                    skip = True
            if skip:
                continue
            if value.grad_req != 'null':
                for arr in value.list_grad():
                    arr *= args.layerwise_decay**(num_layers - i - 1)


def train():
    """Training function."""
    segment = 'train'
    log.info('Loading %s data...', segment)
    # Note that for XLNet, the authors always use squad2 dataset for training
    train_data = SQuAD(segment, version='2.0')
    if args.debug:
        sampled_data = [train_data[i] for i in range(100)]
        train_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in Train data: %s', len(train_data))

    train_data_features = preprocess_dataset(
        tokenizer, train_data, vocab=vocab, max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride, num_workers=args.num_workers,
        max_query_length=args.max_query_length, load_from_pickle=args.load_pickle,
        feature_file=args.train_dataset_file)

    train_data_input = convert_full_features_to_input_features(train_data_features)
    log.info('The number of examples after preprocessing: %s', len(train_data_input))

    train_dataloader = mx.gluon.data.DataLoader(train_data_input, batchify_fn=batchify_fn,
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

    num_train_examples = len(train_data_input)
    step_size = args.batch_size * args.accumulate if args.accumulate else args.batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    epoch_number = args.epochs
    if args.training_steps:
        num_train_steps = args.training_steps
        epoch_number = 100000

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
                    _, inputs, token_types, valid_length, p_mask, start_label, end_label, is_impossible = splited_data  # pylint: disable=line-too-long
                    valid_length = valid_length.astype('float32')
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
                    batch_loss_sep.append(out_sep)
                    batch_loss.append(ls)
                    if args.accumulate:
                        ls = ls / args.accumulate
                    ls.backward()
            # update
            if not args.accumulate or (batch_id + 1) % args.accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                _apply_gradient_decay()
                trainer.update(1, ignore_stale_grad=True)

                step_loss_sep_tmp = np.array(
                    [[span_ls.mean().asscalar(),
                      cls_ls.mean().asscalar()] for span_ls, cls_ls in batch_loss_sep])
                step_loss_sep_tmp = list(np.sum(step_loss_sep_tmp, axis=0))
                step_loss_span += step_loss_sep_tmp[0] / len(ctx)
                step_loss_cls += step_loss_sep_tmp[1] / len(ctx)

            step_loss += sum([ls.asscalar() for ls in batch_loss])
            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                log.info(
                    'Epoch: %d, Batch: %d/%d, Loss=%.4f, lr=%.7f '
                    'Time cost=%.1f Thoughput=%.2f samples/s', epoch_id + 1, batch_id + 1,
                    len(train_dataloader), step_loss / log_interval, trainer.learning_rate,
                    toc - tic, log_num / (toc - tic))
                log.info('span_loss: %.4f, cls_loss: %.4f', step_loss_span / log_interval,
                         step_loss_cls / log_interval)

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
        version_prefix = 'squad2' if args.version_2 else 'squad1'
        ckpt_name = 'model_{}_{}_{}.params'.format(args.model, version_prefix, epoch_id + 1)
        params_saved = os.path.join(args.output_dir, ckpt_name)
        nlp.utils.save_parameters(net, params_saved)
        log.info('params saved in: %s', params_saved)


RawResultExtended = collections.namedtuple(
    'RawResultExtended',
    ['start_top_log_probs', 'start_top_index', 'end_top_log_probs', 'end_top_index', 'cls_logits'])


def evaluate():
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

    dev_data_features = preprocess_dataset(
        tokenizer, dev_data, vocab=vocab, max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride, num_workers=args.num_workers,
        max_query_length=args.max_query_length, load_from_pickle=args.load_pickle,
        feature_file=args.dev_dataset_file)

    dev_data_input = convert_full_features_to_input_features(dev_data_features)
    log.info('The number of examples after preprocessing: %d', len(dev_data_input))

    dev_dataloader = mx.gluon.data.DataLoader(dev_data_input, batchify_fn=batchify_fn,
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
                result = RawResultExtended(start_top_log_probs=outputs[0][c].asnumpy().tolist(),
                                           start_top_index=outputs[1][c].asnumpy().tolist(),
                                           end_top_log_probs=outputs[2][c].asnumpy().tolist(),
                                           end_top_index=outputs[3][c].asnumpy().tolist(),
                                           cls_logits=outputs[4][c].asnumpy().tolist())
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
    for features in dev_data_features:
        results = all_results[features[0].example_id]
        example_qas_id = features[0].qas_id
        score_diff, best_non_null_entry, nbest_json = predict_extended(
            features=features, results=results, n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length, start_n_top=args.start_top_n,
            end_n_top=args.end_top_n)
        scores_diff_json[example_qas_id] = score_diff
        all_predictions[example_qas_id] = best_non_null_entry
        all_nbest_json[example_qas_id] = nbest_json

    output_prediction_file = os.path.join(args.output_dir, 'predictions.json')
    output_nbest_file = os.path.join(args.output_dir, 'nbest_predictions.json')
    output_null_log_odds_file = os.path.join(args.output_dir, 'null_odds.json')

    with open(output_prediction_file, 'w') as writer:
        writer.write(json.dumps(all_predictions, indent=4) + '\n')
    with open(output_nbest_file, 'w') as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + '\n')
    with open(output_null_log_odds_file, 'w') as writer:
        writer.write(json.dumps(scores_diff_json, indent=4) + '\n')

    if os.path.exists(sys.path[0] + '/evaluate-v2.0.py'):
        arguments = [
            dev_data_path, output_prediction_file, '--na-prob-thresh',
            str(args.null_score_diff_threshold)
        ]
        if args.version_2:
            arguments += ['--na-prob-file', output_null_log_odds_file]
        subprocess.call([sys.executable, sys.path[0] + '/evaluate-v2.0.py'] + arguments)
    else:
        log.info('Please download evaluate-v2.0.py to get evaluation results for SQuAD. '
                 'Check index.rst for the detail.')


if __name__ == '__main__':
    if not args.only_predict:
        train()
        evaluate()
    else:
        evaluate()
