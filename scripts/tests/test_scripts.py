import os
import subprocess
import time

import pytest

from ..machine_translation.dataset import TOY


@pytest.mark.remote_required
def test_toy():
    # Test toy dataset
    train_en_de = TOY(segment='train', root='tests/data/translation_test')
    val_en_de = TOY(segment='val', root='tests/data/translation_test')
    test_en_de = TOY(segment='test', root='tests/data/translation_test')
    assert len(train_en_de) == 30
    assert len(val_en_de) == 30
    assert len(test_en_de) == 30
    en_vocab, de_vocab = train_en_de.src_vocab, train_en_de.tgt_vocab
    assert len(en_vocab) == 358
    assert len(de_vocab) == 381
    train_de_en = TOY(segment='train', src_lang='de', tgt_lang='en',
                      root='tests/data/translation_test')
    de_vocab, en_vocab = train_de_en.src_vocab, train_de_en.tgt_vocab
    assert len(en_vocab) == 358
    assert len(de_vocab) == 381
    for i in range(10):
        lhs = train_en_de[i]
        rhs = train_de_en[i]
        assert lhs[0] == rhs[1] and rhs[0] == lhs[1]
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.parametrize('model', ['skipgram', 'cbow'])
@pytest.mark.parametrize('fasttext', [True, False])
def test_skipgram_cbow(model, fasttext):
    cmd = [
        'python', './scripts/word_embeddings/train_sg_cbow.py', '--gpu', '0',
        '--epochs', '2', '--model', model, '--data', 'toy', '--batch-size',
        '64']
    if fasttext:
        cmd += ['--ngram-buckets', '1000']
    else:
        cmd += ['--ngram-buckets', '0']
    subprocess.check_call(cmd)
    time.sleep(5)


def test_glove():
    path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    vocab = os.path.join(path, 'word_embeddings/glove/vocab.txt')
    cooccur = os.path.join(path, 'word_embeddings/glove/cooccurrences.npz')
    cmd = [
        'python', './scripts/word_embeddings/train_glove.py', cooccur, vocab,
        '--batch-size', '2', '--epochs', '2']
    subprocess.check_call(cmd)
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.parametrize('fasttextloadngrams', [True, False])
def test_embedding_evaluate_pretrained(fasttextloadngrams):
    cmd = [
        'python', './scripts/word_embeddings/evaluate_pretrained.py',
        '--embedding-name', 'fasttext', '--embedding-source', 'wiki.simple',
        '--gpu', '0'
    ]
    if fasttextloadngrams:
        cmd.append('--fasttext-load-ngrams')

    subprocess.check_call(cmd)
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
@pytest.mark.parametrize('evaluateanalogies', [True, False])
@pytest.mark.parametrize('maxvocabsize', [None, 16])
def test_embedding_evaluate_from_path(evaluateanalogies, maxvocabsize):
    path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    path = os.path.join(
        path, '../../tests/unittest/train/test_embedding/lorem_ipsum.bin')
    cmd = [
        'python', './scripts/word_embeddings/evaluate_pretrained.py',
        '--embedding-path', path, '--gpu', '0']
    if evaluateanalogies:
        cmd += ['--analogy-datasets', 'GoogleAnalogyTestSet']
    else:
        cmd += ['--analogy-datasets']
    if maxvocabsize is not None:
        cmd += ['--analogy-max-vocab-size', str(maxvocabsize)]
    subprocess.check_call(cmd)
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
def test_sentiment_analysis_finetune():
    process = subprocess.check_call(['python', './scripts/sentiment_analysis/finetune_lm.py',
                                     '--gpu', '0', '--batch_size', '32', '--bucket_type', 'fixed',
                                     '--epochs', '1', '--dropout', '0', '--no_pretrained',
                                     '--lr', '0.005', '--valid_ratio', '0.1',
                                     '--save-prefix', 'imdb_lstm_200'])
    time.sleep(5)
    process = subprocess.check_call(['python', './scripts/sentiment_analysis/finetune_lm.py',
                                     '--gpu', '0', '--batch_size', '32', '--bucket_type', 'fixed',
                                     '--epochs', '1', '--dropout', '0',
                                     '--lr', '0.005', '--valid_ratio', '0.1',
                                     '--save-prefix', 'imdb_lstm_200'])
    time.sleep(5)

@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
def test_sentiment_analysis_textcnn():
    process = subprocess.check_call(['python', './scripts/sentiment_analysis/sentiment_analysis_cnn.py',
                                     '--gpu', '0', '--batch_size', '50', '--epochs', '1',
                                     '--dropout', '0.5', '--lr', '0.0001', '--model_mode', 'rand',
                                     '--data_name', 'MR', '--save-prefix', 'sa-model'])
    time.sleep(5)

@pytest.mark.remote_required
def test_sampling():
    process = subprocess.check_call(['python', './scripts/text_generation/sequence_sampling.py',
                                     '--use-beam-search', '--bos', 'I love it', '--beam_size', '2',
                                     '--print_num', '1'])
    time.sleep(5)
    process = subprocess.check_call(['python', './scripts/text_generation/sequence_sampling.py',
                                     '--use-sampling', '--bos', 'I love it', '--beam_size', '2',
                                     '--print_num', '1', '--temperature', '1.0'])
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
def test_gnmt():
    process = subprocess.check_call(['python', './scripts/machine_translation/train_gnmt.py', '--dataset', 'TOY',
                                     '--src_lang', 'en', '--tgt_lang', 'de', '--batch_size', '32',
                                     '--optimizer', 'adam', '--lr', '0.0025', '--save_dir', 'test',
                                     '--epochs', '1', '--gpu', '0', '--num_buckets', '5',
                                     '--num_hidden', '64', '--num_layers', '2'])
    time.sleep(5)


@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.gpu
def test_transformer():
    process = subprocess.check_call(['python', './scripts/machine_translation/train_transformer.py',
                                     '--dataset', 'TOY', '--src_lang', 'en', '--tgt_lang', 'de',
                                     '--batch_size', '32', '--optimizer', 'adam',
                                     '--num_accumulated', '1', '--lr', '1.0',
                                     '--warmup_steps', '2000', '--save_dir', 'test',
                                     '--epochs', '1', '--gpus', '0', '--scaled', '--average_start',
                                     '1', '--num_buckets', '5', '--bleu', 'tweaked', '--num_units',
                                     '32', '--hidden_size', '64', '--num_layers', '2',
                                     '--num_heads', '4', '--test_batch_size', '32'])
    process = subprocess.check_call(['python', './scripts/machine_translation/train_transformer.py',
                                     '--dataset', 'TOY', '--src_lang', 'en', '--tgt_lang', 'de',
                                     '--batch_size', '32', '--optimizer', 'adam',
                                     '--num_accumulated', '1', '--lr', '1.0',
                                     '--warmup_steps', '2000', '--save_dir', 'test',
                                     '--epochs', '1', '--gpus', '0', '--scaled', '--average_start',
                                     '1', '--num_buckets', '5', '--bleu', '13a', '--num_units',
                                     '32', '--hidden_size', '64', '--num_layers', '2',
                                     '--num_heads', '4', '--test_batch_size', '32'])
    time.sleep(5)
