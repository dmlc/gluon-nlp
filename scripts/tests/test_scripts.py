import subprocess

import pytest
import time

from ..machine_translation.dataset import TOY


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
def test_embedding():
    process = subprocess.check_call([
        'python', './scripts/word_embeddings/train_fasttext.py', '--gpu', '0',
        '--epochs', '1', '--optimizer', 'sgd', '--ngram-buckets', '100',
        '--max-vocab-size', '100', '--batch-size', '64'
    ])
    time.sleep(5)


@pytest.mark.serial
def test_embedding_evaluate_pretrained():
    process = subprocess.check_call([
        'python', './scripts/word_embeddings/evaluate_pretrained.py',
        '--embedding-name', 'fasttext', '--embedding-source', 'wiki.simple'
    ])
    time.sleep(5)


@pytest.mark.serial
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
def test_sentiment_analysis_textcnn():
    process = subprocess.check_call(['python', './scripts/sentiment_analysis/sentiment_analysis_cnn.py',
                                     '--gpu', '0', '--batch_size', '50', '--epochs', '1',
                                     '--dropout', '0.5', '--lr', '0.0001', '--model_mode', 'rand',
                                     '--data_name', 'MR', '--save-prefix', 'sa-model'])
    time.sleep(5)

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
def test_gnmt():
    process = subprocess.check_call(['python', './scripts/machine_translation/train_gnmt.py', '--dataset', 'TOY',
                                     '--src_lang', 'en', '--tgt_lang', 'de', '--batch_size', '32',
                                     '--optimizer', 'adam', '--lr', '0.0025', '--save_dir', 'test',
                                     '--epochs', '1', '--gpu', '0', '--num_buckets', '5',
                                     '--num_hidden', '64', '--num_layers', '2'])
    time.sleep(5)


@pytest.mark.serial
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
