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
"""
Export and deploy the BERT Model for Deployment (testing with Validation datasets)
====================================

This script exports the BERT model to a hybrid model serialized as a symbol.json file,
which is suitable for deployment, or use with MXNet Module API.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

import argparse
import collections
import json
import logging
import warnings
from functools import partial
import os
import io
import time

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import get_model, BERTClassifier
from gluonnlp.data import SQuAD
from gluonnlp.data.classification import get_task
from model.qa import BertForQA
from finetune_squad import preprocess_dataset as qa_preprocess_data
from finetune_classifier import convert_examples_to_features as classifier_examples2features
from bert_qa_evaluate import get_F1_EM, predict, PredResult

nlp.utils.check_version('0.9')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export hybrid BERT base model.')

    parser.add_argument('--model_parameters',
                        type=str,
                        default=None,
                        help='The model parameter file saved from training.')

    parser.add_argument('--bert_model',
                        type=str,
                        default='bert_12_768_12',
                        choices=['bert_12_768_12', 'bert_24_1024_16'],
                        help='BERT model name. Options are "bert_12_768_12" and "bert_24_1024_16"')

    parser.add_argument('--task',
                        type=str,
                        choices=['QA', 'embedding', 'MRPC', 'QQP', 'QNLI', 'RTE', 'STS-B', 'CoLA',
                                 'MNLI', 'WNLI', 'SST', 'XNLI', 'LCQMC', 'ChnSentiCorp'],
                        help='In Classification:'
                        'The name of the task to fine-tune. Choices are QA, embedding, '
                        'MRPC, QQP, QNLI, RTE, STS-B, CoLA, MNLI, WNLI, SST, XNLI, LCQMC, '
                        'ChnSentiCorp')

    parser.add_argument('--dataset_name',
                        type=str,
                        default='book_corpus_wiki_en_uncased',
                        choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                                 'wiki_multilingual_uncased', 'wiki_multilingual_cased',
                                 'wiki_cn_cased'],
                        help='BERT dataset name. Options include '
                             '"book_corpus_wiki_en_uncased", "book_corpus_wiki_en_cased", '
                             '"wiki_multilingual_uncased", "wiki_multilingual_cased", '
                             '"wiki_cn_cased"')

    parser.add_argument('--output_dir',
                        type=str,
                        default='./output_dir',
                        help='The directory where the exported model symbol will be created. '
                             'The default is ./output_dir')

    parser.add_argument('--exported_model',
                        type=str,
                        default=None,
                        help='Prefix path of exported model:'
                        'Should be prefix for -symbol.json / -0000.params files')

    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='Test batch size. default is 128')

    parser.add_argument('--seq_length',
                        type=int,
                        default=128,
                        help='The maximum total input sequence length after WordPiece tokenization.'
                             'Sequences longer than this needs to be truncated, and sequences '
                             'shorter than this needs to be padded. Default is 128')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='The dropout probability for the classification/regression head.')

    parser.add_argument('--gpu',
                        type=int,
                        default=None,
                        help='Id of the gpu to use. Set it to empty means to use cpu.')

    parser.add_argument('--only_infer',
                        action='store_true',
                        help='if set, it does not export the model again.')

    parser.add_argument('--dtype',
                        type=str,
                        default='float32',
                        help='Data type used for training. Either float32 or float16')

    parser.add_argument('--custom_pass',
                        type=str,
                        default=None,
                        help='Specify a custom graph pass for the network (library),'
                        'allowing to customize the graph')

    parser.add_argument('--max_iters',
                        type=int,
                        default=None,
                        help='If set, it runs the maximum number of iterations specified')

    parser.add_argument('--check_accuracy',
                        action='store_true',
                        help='If set, it will check accuracy')

    # Specific for QA
    parser.add_argument('--QA_version_2',
                        action='store_true',
                        help='In Question-Answering:'
                        'SQuAD examples whether contain some that do not have an answer.')

    parser.add_argument('--QA_n_best_size',
                        type=int,
                        default=20,
                        help='In Question-Answering:'
                        'The total number of n-best predictions to generate in the '
                        'nbest_predictions.json output file. default is 20')

    parser.add_argument('--QA_max_answer_length',
                        type=int,
                        default=30,
                        help='In Question-Answering:'
                        'The maximum length of an answer that can be generated. This is needed '
                        'because the start and end predictions are not conditioned on one another.'
                        ' default is 30')

    parser.add_argument('--QA_doc_stride',
                        type=int,
                        default=128,
                        help='In Question-Answering:'
                        'When splitting up a long document into chunks, how much stride to '
                        'take between chunks. default is 128')

    parser.add_argument('--QA_max_query_length',
                        type=int,
                        default=64,
                        help='In Question-Answering:'
                        'The maximum number of tokens for the question. Questions longer than '
                        'this will be truncated to this length. default is 64')

    parser.add_argument('--QA_null_score_diff_threshold',
                        type=float,
                        default=0.0,
                        help='In Question-Answering:'
                        'If null_score - best_non_null is greater than the threshold predict null.'
                        'Typical values are between -1.0 and -5.0. default is 0.0')

    # specific for embedding
    parser.add_argument('--oov_way', type=str, default='avg',
                        help='how to handle subword embeddings\n'
                             'avg: average all subword embeddings to represent the original token\n'
                             'sum: sum all subword embeddings to represent the original token\n'
                             'last: use last subword embeddings to represent the original token\n')

    args = parser.parse_args()

    # create output dir
    output_dir = args.output_dir
    nlp.utils.mkdir(output_dir)

    #set context and type
    if args.gpu is not None:
        ctx = mx.gpu(args.gpu)
    else:
        ctx = mx.cpu()
    dtype = args.dtype

    ###############################################################################
    #                                Logging                                      #
    ###############################################################################

    log = logging.getLogger('gluonnlp')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s',
                                  datefmt='%H:%M:%S')
    fh = logging.FileHandler(os.path.join(args.output_dir, 'hybrid_export_bert.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    log.addHandler(console)
    log.addHandler(fh)
    log.info(args)

    ###############################################################################
    #                              Hybridize the model                            #
    ###############################################################################
    export_ctx = mx.cpu()
    seq_length = args.seq_length
    do_lower_case = 'uncased' in args.dataset_name

    if args.task == 'QA':
        bert, vocab = get_model(
            name=args.bert_model,
            dataset_name=args.dataset_name,
            pretrained=False,
            use_pooler=False,
            use_decoder=False,
            use_classifier=False,
            ctx=export_ctx)
        net = BertForQA(bert)
    elif args.task == 'embedding':
        bert, vocab = get_model(
            name=args.bert_model,
            dataset_name=args.dataset_name,
            pretrained=True,
            use_pooler=False,
            use_decoder=False,
            use_classifier=False,
            ctx=export_ctx)
        net = bert
    else:
        specific_task = get_task(args.task)
        do_regression = not specific_task.class_labels
        if do_regression:
            bert, vocab = get_model(
                name=args.bert_model,
                dataset_name=args.dataset_name,
                pretrained=False,
                use_pooler=True,
                use_decoder=False,
                use_classifier=False,
                ctx=export_ctx)
            net = BERTClassifier(bert, num_classes=1, dropout=args.dropout)
        else:
            # classification task
            bert, vocab = get_model(
                name=args.bert_model,
                dataset_name=args.dataset_name,
                pretrained=False,
                use_pooler=True,
                use_decoder=False,
                use_classifier=False)
            num_classes = len(specific_task.class_labels)
            net = BERTClassifier(bert, num_classes=num_classes, dropout=args.dropout)

    if args.model_parameters and args.task != 'embedding':
        net.load_parameters(args.model_parameters, ctx=export_ctx)
    elif args.task != 'embedding':
        net.initialize(ctx=export_ctx)
        warnings.warn('--model_parameters is not provided. The parameter checkpoint (.params) '
                      'file will be created based on default parameter initialization.')

    net.hybridize(static_alloc=True, static_shape=True)
    test_batch_size = args.test_batch_size

###############################################################################
#                              Export the model                               #
###############################################################################
def export(prefix):
    """Export the model."""
    log.info('Exporting the model ... ')

    # dummy input data
    inputs = mx.nd.arange(test_batch_size * seq_length)
    inputs = inputs.reshape(shape=(test_batch_size, seq_length))
    token_types = mx.nd.zeros_like(inputs)
    valid_length = mx.nd.arange(test_batch_size)
    batch = inputs, token_types, valid_length
    inputs, token_types, valid_length = batch

    net(inputs.as_in_context(export_ctx),
        token_types.as_in_context(export_ctx),
        valid_length.as_in_context(export_ctx))
    net.export(prefix, epoch=0)
    assert os.path.isfile(prefix + '-symbol.json')
    assert os.path.isfile(prefix + '-0000.params')

    if args.custom_pass is not None:
        # load library
        libpath = os.path.abspath(args.custom_pass)
        mx.library.load(libpath)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)

        arg_array = arg_params
        arg_array['data0'] = mx.nd.ones((test_batch_size, seq_length), dtype='float32')
        arg_array['data1'] = mx.nd.ones((test_batch_size, seq_length), dtype='float32')
        arg_array['data2'] = mx.nd.ones((test_batch_size, ), dtype='float32')
        custom_sym = sym.optimize_for('custom_pass', arg_array, aux_params)

        nheads = 12
        if args.bert_model == 'bert_24_1024_16':
            nheads = 24
        for i in range(nheads):
            basename = 'bertencoder0_transformer' + str(i) + '_dotproductselfattentioncell0'
            arg_array.pop(basename + '_query_weight')
            arg_array.pop(basename + '_key_weight')
            arg_array.pop(basename + '_value_weight')
            arg_array.pop(basename + '_query_bias')
            arg_array.pop(basename + '_key_bias')
            arg_array.pop(basename + '_value_bias')
        arg_array.pop('data0')
        arg_array.pop('data1')
        arg_array.pop('data2')

        mx.model.save_checkpoint(prefix, 0, custom_sym, arg_params, aux_params)

# Function to preprocess dataset to test, which depends on the task
def preprocess_data(tokenizer, task):
    """Preprocess dataset to test."""
    log.info('Loading dev data...')
    if task == 'QA':
        # question_answering
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=seq_length),
            nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=seq_length),
            nlp.data.batchify.Stack('float32'),
            nlp.data.batchify.Stack('float32'),
            nlp.data.batchify.Stack('float32'))
        if args.QA_version_2:
            dev_data = SQuAD('dev', version='2.0')
        else:
            dev_data = SQuAD('dev', version='1.1')
        dev_dataset = qa_preprocess_data(tokenizer,
                                         dev_data,
                                         max_seq_length=seq_length,
                                         doc_stride=args.QA_doc_stride,
                                         max_query_length=args.QA_max_query_length,
                                         input_features=False)
        dev_data_transform = qa_preprocess_data(tokenizer,
                                                dev_data,
                                                max_seq_length=seq_length,
                                                doc_stride=args.QA_doc_stride,
                                                max_query_length=args.QA_max_query_length,
                                                input_features=True)
        dev_dataloader = mx.gluon.data.DataLoader(dev_data_transform,
                                                  batchify_fn=batchify_fn,
                                                  num_workers=4,
                                                  batch_size=test_batch_size,
                                                  shuffle=False,
                                                  last_batch='keep')
        return dev_dataloader, len(dev_data_transform), dev_dataset

    else:
        # classification / regression
        classification_task = get_task(task)

        label_dtype = 'int32' if classification_task.class_labels else 'float32'
        truncate_length = seq_length - 3 if classification_task.is_pair else seq_length - 2
        trans = partial(classifier_examples2features, tokenizer=tokenizer,
                        truncate_length=truncate_length,
                        cls_token=vocab.cls_token,
                        sep_token=vocab.sep_token,
                        class_labels=classification_task.class_labels,
                        label_alias=classification_task.label_alias, vocab=vocab)

        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token],
                                  round_to=seq_length), # input
            nlp.data.batchify.Pad(axis=0, pad_val=0, round_to=seq_length),  # segment
            nlp.data.batchify.Stack(),  # length
            nlp.data.batchify.Stack(label_dtype))  # label

        # data dev. For MNLI, more than one dev set is available
        dev_tsv = classification_task.dataset_dev()
        dev_tsv_list = dev_tsv if isinstance(dev_tsv, list) else [dev_tsv]
        loader_dev_list = []
        nsamples = 0
        for segment, data in dev_tsv_list:
            data_dev = mx.gluon.data.SimpleDataset(list(map(trans, data)))
            nsamples = nsamples + len(data_dev)
            loader_dev = mx.gluon.data.DataLoader(data_dev,
                                                  batchify_fn=batchify_fn,
                                                  num_workers=4,
                                                  batch_size=test_batch_size,
                                                  shuffle=False,
                                                  last_batch='keep')
            loader_dev_list.append((segment, loader_dev))
        return loader_dev_list, nsamples, None

# Function to calculate final accuracy and print it out. It also save predictions within a file
def compute_accuracy_save_results(task, all_results, SQuAD_dataset=None, segment=None, metric=None):
    """Compute accuracy and save predictions."""
    all_predictions = collections.OrderedDict()
    if task == 'QA':
        assert SQuAD_dataset is not None
        if args.QA_version_2:
            dev_data = SQuAD('dev', version='2.0')
        else:
            dev_data = SQuAD('dev', version='1.1')
        for features in SQuAD_dataset:
            results = all_results[features[0].example_id]
            example_qas_id = features[0].qas_id
            prediction, _ = predict(
                features=features,
                results=results,
                tokenizer=nlp.data.BERTBasicTokenizer(lower=do_lower_case),
                max_answer_length=args.QA_max_answer_length,
                null_score_diff_threshold=args.QA_null_score_diff_threshold,
                n_best_size=args.QA_n_best_size,
                version_2=args.QA_version_2)
            all_predictions[example_qas_id] = prediction
        if args.QA_version_2:
            log.info('Please run evaluate-v2.0.py to get evaluation results for SQuAD 2.0')
        else:
            F1_EM = get_F1_EM(dev_data, all_predictions)
            log.info(F1_EM)
        # save results
        with io.open(os.path.join(output_dir, task + '-predictions.json'),
                     'w', encoding='utf-8') as fout:
            data = json.dumps(all_predictions, ensure_ascii=False)
            fout.write(data)

    elif task == 'embedding':
        final_results = []
        padding_idx, cls_idx, sep_idx = None, None, None
        if vocab.padding_token:
            padding_idx = vocab[vocab.padding_token]
        if vocab.cls_token:
            cls_idx = vocab[vocab.cls_token]
        if vocab.sep_token:
            sep_idx = vocab[vocab.sep_token]
        for token_ids, sequence_outputs in all_results:
            token_ids = token_ids.astype(int)
            tokens = []
            tensors = []
            oov_len = 1
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                # [PAD] token, sequence is finished.
                if padding_idx and token_id == padding_idx:
                    break
                # [CLS], [SEP]
                if cls_idx and token_id == cls_idx:
                    continue
                if sep_idx and token_id == sep_idx:
                    continue
                token = vocab.idx_to_token[token_id]
                tokenizer = nlp.data.BERTTokenizer(vocab, lower=do_lower_case)
                if not tokenizer.is_first_subword(token):
                    tokens.append(token)
                    if args.oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if args.oov_way == 'avg':
                        oov_len += 1
                else:  # iv, avg last oov
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            final_results.append((tokens, tensors))

        with io.open(os.path.join(output_dir, task + '-output.tsv'),
                     'w', encoding='utf-8') as fout:
            for embeddings in final_results:
                sent, tokens_embedding = embeddings
                fout.write(u'Text: \t%s\n' % (str(sent)))
                fout.write(u'Tokens embedding: \t%s\n\n' % (str(tokens_embedding)))

    else:
        # classification / regression
        assert segment is not None
        assert metric is not None
        specific_task = get_task(task)
        metric_nm, metric_val = metric.get()
        if not isinstance(metric_nm, list):
            metric_nm, metric_val = [metric_nm], [metric_val]
        metric_str = 'validation metrics: ' + ', '.join([i + ':%.4f' for i in metric_nm])
        log.info(metric_str, *metric_val)
        # save results
        final_results = []
        if not specific_task.class_labels:
            # regression task
            for result in all_results:
                for probs in result.asnumpy().reshape(-1).tolist():
                    final_results.append('{:.3f}'.format(probs))
        else:
            # classification task
            for result in all_results:
                indices = mx.nd.topk(result, k=1, ret_typ='indices', dtype='int32').asnumpy()
                for index in indices:
                    final_results.append(specific_task.class_labels[int(index)])
        with io.open(os.path.join(output_dir, task + '-' + segment + '-predictions.tsv'),
                     'w', encoding='utf-8') as fout:
            fout.write(u'index\tprediction\n')
            for i, pred in enumerate(final_results):
                fout.write(u'%d\t%s\n\n' % (i, str(pred)))

###############################################################################
#                             Perform inference                               #
###############################################################################
def infer(prefix, task):
    """Perform inference."""
    assert os.path.isfile(prefix + '-symbol.json')
    assert os.path.isfile(prefix + '-0000.params')

     # import with SymbolBlock. Alternatively, you can use Module.load APIs.
    imported_net = mx.gluon.nn.SymbolBlock.imports(prefix + '-symbol.json',
                                                   ['data0', 'data1', 'data2'],
                                                   prefix + '-0000.params',
                                                   ctx=ctx)
    imported_net.hybridize(static_alloc=True, static_shape=True)
    if dtype == 'float16':
        imported_net.cast('float16')
    tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=do_lower_case)

    num_warmup = 2

    if task == 'QA':
        dataloader, _, SQuAD_dataset = preprocess_data(tokenizer, task)
        # run warmup iterations
        for data in dataloader:
            example_ids, token_ids, token_types, valid_length, _, _ = data
            out = imported_net(token_ids.as_in_context(ctx),
                               token_types.as_in_context(ctx),
                               valid_length.as_in_context(ctx).astype(dtype))
            output = mx.nd.split(out, axis=2, num_outputs=2)
            pred_start = output[0].reshape((0, -3)).asnumpy()
            pred_end = output[1].reshape((0, -3)).asnumpy()
            num_warmup -= 1
            if not num_warmup:
                break
        # run forward inference
        log.info('Start inference ... ')
        total_iters = 0
        total_samples = 0
        total_latency_time = 0.0
        all_results = collections.defaultdict(list)
        tic = time.time()
        for data in dataloader:
            example_ids, token_ids, token_types, valid_length, _, _ = data
            tic_latency = time.time()
            out = imported_net(token_ids.as_in_context(ctx),
                               token_types.as_in_context(ctx),
                               valid_length.as_in_context(ctx).astype(dtype))
            output = mx.nd.split(out, axis=2, num_outputs=2)
            pred_start = output[0].reshape((0, -3)).asnumpy()
            pred_end = output[1].reshape((0, -3)).asnumpy()
            toc_latency = time.time()
            total_latency_time += (toc_latency - tic_latency)
            total_iters += 1
            total_samples += len(token_ids)
            if args.check_accuracy:
                example_ids = example_ids.asnumpy().tolist()
                for example_id, start, end in zip(example_ids, pred_start, pred_end):
                    all_results[example_id].append(PredResult(start=start, end=end))
            if args.max_iters and total_iters >= args.max_iters:
                break
        mx.nd.waitall()
        toc = time.time()
        log.info('BatchSize={}, NumberIterations={}:  '.format(test_batch_size, total_iters))
        log.info('Throughput={:.2f} samples/s, Average Latency={:.4f} ms'
                 .format(total_samples / (toc - tic), (total_latency_time / total_iters) * 1000))
        if args.check_accuracy:
            compute_accuracy_save_results(task, all_results, SQuAD_dataset=SQuAD_dataset)

    elif task == 'embedding':
        # Uses SST dataset as example
        dataloader_list, _, _ = preprocess_data(tokenizer, 'SST')
        _, dataloader = dataloader_list[0]
        # run warmup iterations
        for data in dataloader:
            token_ids, token_types, valid_length, _ = data
            sequence_outputs = imported_net(token_ids.as_in_context(ctx),
                                            token_types.as_in_context(ctx),
                                            valid_length.as_in_context(ctx).astype(dtype))
            sequence_outputs.asnumpy()
            num_warmup -= 1
            if not num_warmup:
                break
        # run forward inference
        log.info('Start inference ... ')
        total_iters = 0
        total_samples = 0
        total_latency_time = 0.0
        all_results = []
        tic = time.time()
        for data in dataloader:
            token_ids, token_types, valid_length, _ = data
            tic_latency = time.time()
            sequence_outputs = imported_net(token_ids.as_in_context(ctx),
                                            token_types.as_in_context(ctx),
                                            valid_length.as_in_context(ctx).astype(dtype))
            sequence_outputs.asnumpy()
            toc_latency = time.time()
            total_latency_time += (toc_latency - tic_latency)
            total_iters += 1
            total_samples += len(token_ids)
            if args.check_accuracy:
                for token_id, sequence_output in zip(token_ids.asnumpy(),
                                                     sequence_outputs.asnumpy()):
                    all_results.append((token_id, sequence_output))
            if args.max_iters and total_iters >= args.max_iters:
                break
        mx.nd.waitall()
        toc = time.time()
        log.info('BatchSize={}, NumberIterations={}:  '.format(test_batch_size, total_iters))
        log.info('Throughput={:.2f} samples/s, Average Latency={:.4f} ms'
                 .format(total_samples / (toc - tic), (total_latency_time / total_iters) * 1000))
        if args.check_accuracy:
            compute_accuracy_save_results(task, all_results)

    else:
        # classification / regression task
        dataloader_list, _, _ = preprocess_data(tokenizer, task)
        specific_task = get_task(task)
        metric = specific_task.metrics
        # run warmup iterations
        _, dataloader = dataloader_list[0]
        for data in dataloader:
            token_ids, token_types, valid_length, label = data
            out = imported_net(token_ids.as_in_context(ctx),
                               token_types.as_in_context(ctx),
                               valid_length.as_in_context(ctx).astype(dtype))
            out.asnumpy()
            num_warmup -= 1
            if not num_warmup:
                break
        # run forward inference
        for segment, dataloader in dataloader_list:
            log.info('Start inference ... ')
            total_iters = 0
            total_samples = 0
            total_latency_time = 0.0
            all_results = []
            metric.reset()
            tic = time.time()
            for data in dataloader:
                token_ids, token_types, valid_length, label = data
                label = label.as_in_context(ctx)
                tic_latency = time.time()
                out = imported_net(token_ids.as_in_context(ctx),
                                   token_types.as_in_context(ctx),
                                   valid_length.as_in_context(ctx).astype(dtype))
                out.asnumpy()
                toc_latency = time.time()
                total_latency_time += (toc_latency - tic_latency)
                total_iters += 1
                total_samples += len(token_ids)
                if args.check_accuracy:
                    if not do_regression:
                        label = label.reshape((-1))
                    metric.update([label], [out])
                    all_results.append(out)
                if args.max_iters and total_iters >= args.max_iters:
                    break
            mx.nd.waitall()
            toc = time.time()
            log.info('Segment {}'.format(segment))
            log.info('BatchSize={}, NumberIterations={}:  '.format(test_batch_size, total_iters))
            log.info('Throughput={:.2f} samples/s, Average Latency={:.4f} ms'
                     .format(total_samples / (toc - tic),
                             (total_latency_time / total_iters) * 1000))
            if args.check_accuracy:
                compute_accuracy_save_results(task, all_results, segment=segment, metric=metric)

if __name__ == '__main__':
    if args.exported_model:
        prefix = args.exported_model
    else:
        prefix = os.path.join(args.output_dir, args.task)
    if not args.only_infer:
        export(prefix)
    infer(prefix, args.task)
