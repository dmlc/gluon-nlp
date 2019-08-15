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

import os
import subprocess
import sys
import time
import datetime

import pytest
import mxnet as mx

@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize('model', ['skipgram', 'cbow'])
@pytest.mark.parametrize('fasttext', [True, False])
def test_skipgram_cbow(model, fasttext):
    cmd = [
        sys.executable, './scripts/word_embeddings/train_sg_cbow.py', '--gpu', '0',
        '--epochs', '2', '--model', model, '--data', 'toy', '--batch-size',
        '64']
    cmd += ['--similarity-datasets', 'WordSim353']
    cmd += ['--analogy-datasets', 'GoogleAnalogyTestSet']
    if fasttext:
        cmd += ['--ngram-buckets', '1000']
    else:
        cmd += ['--ngram-buckets', '0']
    subprocess.check_call(cmd)
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.integration
def test_glove():
    path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    vocab = os.path.join(path, 'word_embeddings/glove/vocab.txt')
    cooccur = os.path.join(path, 'word_embeddings/glove/cooccurrences.npz')
    cmd = [
        sys.executable, './scripts/word_embeddings/train_glove.py', cooccur, vocab,
        '--batch-size', '2', '--epochs', '2', '--gpu', '0']
    cmd += ['--similarity-datasets', 'WordSim353']
    cmd += ['--analogy-datasets', 'GoogleAnalogyTestSet']
    subprocess.check_call(cmd)
    time.sleep(5)


@pytest.mark.skip_master
@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize('fasttextloadngrams', [True, False])
def test_embedding_evaluate_pretrained(fasttextloadngrams):
    cmd = [
        sys.executable, './scripts/word_embeddings/evaluate_pretrained.py',
        '--embedding-name', 'fasttext', '--embedding-source', 'wiki.simple',
        '--gpu', '0'
    ]
    cmd += ['--similarity-datasets', 'WordSim353']
    cmd += ['--analogy-datasets', 'GoogleAnalogyTestSet']
    if fasttextloadngrams:
        cmd.append('--fasttext-load-ngrams')

    subprocess.check_call(cmd)
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize('evaluateanalogies', [True, False])
@pytest.mark.parametrize('maxvocabsize', [None, 16])
def test_embedding_evaluate_from_path(evaluateanalogies, maxvocabsize):
    path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    path = os.path.join(
        path, '../../tests/unittest/train/test_embedding/lorem_ipsum.bin')
    cmd = [
        sys.executable, './scripts/word_embeddings/evaluate_pretrained.py',
        '--embedding-path', path, '--gpu', '0']
    if evaluateanalogies:
        cmd += ['--similarity-datasets=']
        cmd += ['--analogy-datasets', 'GoogleAnalogyTestSet']
    else:
        cmd += ['--similarity-datasets', 'WordSim353']
        cmd += ['--analogy-datasets=']
    if maxvocabsize is not None:
        cmd += ['--analogy-max-vocab-size', str(maxvocabsize)]
    subprocess.check_call(cmd)
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize('use_pretrained', [True, False])
def test_sentiment_analysis_finetune(use_pretrained):
    args = ['--gpu', '0', '--batch_size', '32', '--bucket_type', 'fixed',
            '--epochs', '1', '--dropout', '0',
            '--lr', '0.005', '--valid_ratio', '0.1',
            '--save-prefix', 'imdb_lstm_200']
    if not use_pretrained:
        args.append('--no_pretrained')
    process = subprocess.check_call([sys.executable, './scripts/sentiment_analysis/finetune_lm.py']+args)
    time.sleep(5)

@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.integration
def test_sentiment_analysis_textcnn():
    process = subprocess.check_call([sys.executable, './scripts/sentiment_analysis/sentiment_analysis_cnn.py',
                                     '--gpu', '0', '--batch_size', '50', '--epochs', '1',
                                     '--dropout', '0.5', '--model_mode', 'rand', '--data_name', 'MR'])
    time.sleep(5)

@pytest.mark.skip_master
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.parametrize('method', ['beam_search', 'sampling'])
def test_sampling(method):
    args = ['--bos', 'I love it', '--beam-size', '2', '--print-num', '1', '--gpu', '0']
    if method == 'beam_search':
        args.insert(0, 'beam-search')
        args.extend(['--k', '50'])
    if method == 'sampling':
        args.insert(0, 'random-sample')
        args.extend(['--temperature', '1.0'])
    process = subprocess.check_call([sys.executable, './scripts/text_generation/sequence_sampling.py']
                                     + args)
    time.sleep(5)


@pytest.mark.skip_master
@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.integration
def test_gnmt():
    process = subprocess.check_call([sys.executable, './scripts/machine_translation/train_gnmt.py', '--dataset', 'TOY',
                                     '--src_lang', 'en', '--tgt_lang', 'de', '--batch_size', '32',
                                     '--optimizer', 'adam', '--lr', '0.0025', '--save_dir', 'test',
                                     '--epochs', '1', '--gpu', '0', '--num_buckets', '5',
                                     '--num_hidden', '64', '--num_layers', '2'])
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize('bleu', ['tweaked', '13a'])
def test_transformer(bleu):
    args = ['--dataset', 'TOY', '--src_lang', 'en', '--tgt_lang', 'de',
            '--batch_size', '32', '--optimizer', 'adam',
            '--num_accumulated', '1', '--lr', '1.0',
            '--warmup_steps', '2000', '--save_dir', 'test',
            '--epochs', '1', '--gpus', '0', '--scaled', '--average_start',
            '1', '--num_buckets', '5', '--bleu', bleu, '--num_units',
            '32', '--hidden_size', '64', '--num_layers', '2',
            '--num_heads', '4', '--test_batch_size', '32']
    process = subprocess.check_call([sys.executable, './scripts/machine_translation/train_transformer.py']
                                    +args)
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize('use_pretrained', [True, False])
def test_bert_embedding(use_pretrained):
    args = ['--gpu', '0', '--model', 'bert_12_768_12',
            '--dataset_name', 'book_corpus_wiki_en_uncased',
            '--max_seq_length', '25', '--batch_size', '256',
            '--oov_way', 'avg', '--sentences', '"is this jacksonville ?"',
            '--verbose']
    if use_pretrained:
        args.extend(['--dtype', 'float32'])
    else:
        args.extend(['--params_path',
                     '~/.mxnet/models/bert_12_768_12_book_corpus_wiki_en_uncased-75cc780f.params'])
    process = subprocess.check_call([sys.executable, './scripts/bert/embedding.py'] + args)
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
def test_pretrain_create():
    # test data creation
    process = subprocess.check_call([sys.executable, './scripts/bert/create_pretraining_data.py',
                                     '--input_file', './scripts/bert/sample_text.txt',
                                     '--output_dir', 'test/bert/data',
                                     '--dataset_name', 'book_corpus_wiki_en_uncased',
                                     '--max_seq_length', '128',
                                     '--max_predictions_per_seq', '20',
                                     '--dupe_factor', '5',
                                     '--whole_word_mask',
                                     '--masked_lm_prob', '0.15',
                                     '--short_seq_prob', '0.1',
                                     '--verbose'])
    time.sleep(3)

@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
def test_pretrain():
    # test data creation
    process = subprocess.check_call([sys.executable, './scripts/bert/create_pretraining_data.py',
                                     '--input_file', './scripts/bert/sample_text.txt',
                                     '--output_dir', 'test/bert/data',
                                     '--dataset_name', 'book_corpus_wiki_en_uncased',
                                     '--max_seq_length', '128',
                                     '--max_predictions_per_seq', '20',
                                     '--dupe_factor', '5',
                                     '--masked_lm_prob', '0.15',
                                     '--short_seq_prob', '0.1',
                                     '--verbose'])
    try:
        # TODO(haibin) update test once MXNet 1.5 is released.
        from mxnet.ndarray.contrib import adamw_update
        arguments = ['--log_interval', '2', '--data_eval', './test/bert/data/*.npz',
                     '--batch_size_eval', '8', '--ckpt_dir', './test/bert/ckpt', '--gpus', '0',
                     '--num_steps', '20', '--num_buckets', '1']
        # test training
        process = subprocess.check_call([sys.executable, './scripts/bert/run_pretraining.py',
                                         '--dtype', 'float32',
                                         '--data', './test/bert/data/*.npz',
                                         '--batch_size', '32',
                                         '--lr', '2e-5',
                                         '--warmup_ratio', '0.5',
                                         '--pretrained'] + arguments)
        # test evaluation
        process = subprocess.check_call([sys.executable, './scripts/bert/run_pretraining.py',
                                         '--dtype', 'float32',
                                         '--pretrained'] + arguments)

        # test mixed precision training and use-avg-len
        from mxnet.ndarray.contrib import mp_adamw_update
        process = subprocess.check_call([sys.executable, './scripts/bert/run_pretraining.py',
                                         '--data', './test/bert/data/*.npz',
                                         '--batch_size', '4096',
                                         '--use_avg_len',
                                         '--lr', '2e-5',
                                         '--warmup_ratio', '0.5',
                                         '--pretrained'] + arguments)
        time.sleep(5)
    except ImportError:
        print("The test expects master branch of MXNet. Skipped now.")


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
def test_pretrain_hvd():
    # test data creation
    process = subprocess.check_call([sys.executable, './scripts/bert/create_pretraining_data.py',
                                     '--input_file', './scripts/bert/sample_text.txt',
                                     '--output_dir', 'test/bert/data',
                                     '--dataset_name', 'book_corpus_wiki_en_uncased',
                                     '--max_seq_length', '128',
                                     '--max_predictions_per_seq', '20',
                                     '--dupe_factor', '5',
                                     '--masked_lm_prob', '0.15',
                                     '--short_seq_prob', '0.1',
                                     '--verbose'])
    try:
        # TODO(haibin) update test once MXNet 1.5 is released.
        from mxnet.ndarray.contrib import adamw_update
        import horovod.mxnet as hvd
        arguments = ['--log_interval', '2',
                     '--batch_size_eval', '8', '--ckpt_dir', './test/bert/ckpt',
                     '--num_steps', '20', '--num_buckets', '1']
        # test training
        process = subprocess.check_call([sys.executable, './scripts/bert/run_pretraining_hvd.py',
                                         '--dtype', 'float32',
                                         '--data', './test/bert/data/*.npz',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--batch_size', '32',
                                         '--lr', '2e-5', '--eval_use_npz',
                                         '--warmup_ratio', '0.5',
                                         '--pretrained'] + arguments)
        # test training with raw data
        process = subprocess.check_call([sys.executable, './scripts/bert/run_pretraining_hvd.py',
                                         '--dtype', 'float32',
                                         '--raw',
                                         '--max_seq_length', '128',
                                         '--max_predictions_per_seq', '20',
                                         '--masked_lm_prob', '0.15',
                                         '--short_seq_prob', '0.1',
                                         '--data', './scripts/bert/sample_text.txt',
                                         '--data_eval', './scripts/bert/sample_text.txt',
                                         '--batch_size', '32',
                                         '--lr', '2e-5',
                                         '--warmup_ratio', '0.5',
                                         '--pretrained'] + arguments)

        # test evaluation
        process = subprocess.check_call([sys.executable, './scripts/bert/run_pretraining_hvd.py',
                                         '--dtype', 'float32',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--eval_use_npz', '--pretrained'] + arguments)

        # test mixed precision training and use-avg-len
        from mxnet.ndarray.contrib import mp_adamw_update
        process = subprocess.check_call([sys.executable, './scripts/bert/run_pretraining_hvd.py',
                                         '--data', './test/bert/data/*.npz',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--batch_size', '4096',
                                         '--use_avg_len',
                                         '--lr', '2e-5',
                                         '--warmup_ratio', '0.5',
                                         '--pretrained'] + arguments)
        time.sleep(5)
    except ImportError:
        print("The test expects master branch of MXNet and Horovod. Skipped now.")

@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
# MNLI inference (multiple dev sets)
# STS-B inference (regression task)
@pytest.mark.parametrize('dataset', ['MNLI', 'STS-B'])
def test_finetune_inference(dataset):
    arguments = ['--log_interval', '100', '--epsilon', '1e-8', '--optimizer',
                 'adam', '--gpu', '0', '--max_len', '80', '--only_inference']
    process = subprocess.check_call([sys.executable, './scripts/bert/finetune_classifier.py',
                                     '--task_name', dataset] + arguments)
    time.sleep(5)

@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['XNLI', 'ChnSentiCorp'])
def test_finetune_chinese_inference(dataset):
    arguments = ['--log_interval', '100', '--epsilon', '1e-8', '--optimizer',
                 'adam', '--gpu', '0', '--max_len', '80', '--only_inference']
    process = subprocess.check_call([sys.executable, './scripts/bert/finetune_classifier.py',
                                     '--task_name', dataset] + arguments)
    time.sleep(5)

@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['WNLI'])
@pytest.mark.parametrize('dtype', ['float32', 'float16'])
def test_finetune_train(dataset, dtype):
    arguments = ['--log_interval', '100', '--epsilon', '1e-8', '--optimizer',
                 'adam', '--gpu', '0', '--epochs', '2', '--dtype', dtype]
    try:
        # TODO(haibin) update test once MXNet 1.5 is released.
        from mxnet.ndarray.contrib import adamw_update
        # WNLI training with bert_adam
        process = subprocess.check_call([sys.executable, './scripts/bert/finetune_classifier.py',
                                         '--task_name', dataset,
                                         '--optimizer', 'bertadam'] + arguments)
    except ImportError:
        # WNLI training with adam
        process = subprocess.check_call([sys.executable, './scripts/bert/finetune_classifier.py',
                                         '--task_name', dataset,
                                         '--optimizer', 'adam'] + arguments)

@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.parametrize('task', ['classification', 'regression', 'question_answering'])
def test_export(task):
    process = subprocess.check_call([sys.executable, './scripts/bert/export/export.py',
                                     '--task', task])

@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.remote_required
@pytest.mark.integration
@pytest.mark.parametrize('sentencepiece', [False, True])
def test_finetune_squad(sentencepiece):
    arguments = ['--optimizer', 'adam', '--batch_size', '12',
                 '--gpu', '0', '--epochs', '2', '--debug']
    if sentencepiece:
        # the downloaded bpe vocab
        url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-682b5d15.bpe'
        f = mx.test_utils.download(url, overwrite=True)
        arguments += ['--sentencepiece', f]

    process = subprocess.check_call([sys.executable, './scripts/bert/finetune_squad.py']
                                    + arguments)
    time.sleep(5)
