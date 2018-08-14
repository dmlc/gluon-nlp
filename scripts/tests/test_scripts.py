import subprocess

import pytest

from ..nmt.dataset import TOY


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


def test_gnmt():
    process = subprocess.check_call(['python', './scripts/nmt/train_gnmt.py', '--dataset', 'TOY',
                                     '--src_lang', 'en', '--tgt_lang', 'de', '--batch_size', '3',
                                     '--optimizer', 'adam', '--lr', '0.0025', '--save_dir', 'test',
                                     '--epochs', '20', '--gpu', '0', '--num_buckets', '5',
                                     '--num_hidden', '64', '--num_layers', '2'])


def test_transformer():
    process = subprocess.check_call(['python', './scripts/nmt/train_transformer.py',
                                     '--dataset', 'TOY', '--src_lang', 'en', '--tgt_lang', 'de',
                                     '--batch_size', '128', '--optimizer', 'adam',
                                     '--num_accumulated', '1', '--lr', '1.0',
                                     '--warmup_steps', '2000', '--save_dir', 'test',
                                     '--epochs', '5', '--gpus', '0', '--scaled', '--average_start',
                                     '1', '--num_buckets', '5', '--bleu', 'tweaked', '--num_units',
                                     '32', '--hidden_size', '64', '--num_layers', '2',
                                     '--num_heads', '4', '--test_batch_size', '128'])
