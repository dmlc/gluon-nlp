import pytest
import collections
import random
import uuid
import os
import numpy as np
from gluonnlp.data.vocab import Vocab


def test_vocab():
    def check_same_vocab(vocab1, vocab2):
        assert vocab1.all_tokens == vocab2.all_tokens
        assert len(vocab1._special_token_kv) == len(vocab2._special_token_kv)
        for k, v in vocab1._special_token_kv.items():
            assert v == vocab2._special_token_kv[k]
            assert getattr(vocab1, k) == getattr(vocab2, k)

    def check_consistency(vocab):
        for i, token in enumerate(vocab.all_tokens):
            assert vocab[token] == i
        if hasattr(vocab, 'unk_token'):
            assert vocab['some1234123dasf'] == vocab[vocab.unk_token]
        assert len(vocab) == len(vocab.all_tokens)
        if len(vocab.all_tokens) > 0:
            random_idx = [random.randint(0, len(vocab.all_tokens) - 1) for _ in range(20)]
            assert vocab.to_tokens(random_idx) == [vocab.all_tokens[i] for i in random_idx]
            assert vocab.to_tokens(np.array(random_idx)) == [vocab.all_tokens[i] for i in random_idx]
            random_tokens = vocab.to_tokens(random_idx)
            assert vocab[random_tokens] == random_idx
            if vocab.has_unk:
                assert vocab[random_tokens + ['213412hadhfk']]\
                       == random_idx + [vocab.unk_id]
            for k, v in vocab.special_tokens_kv.items():
                print(k)
                idx_property = k[:-6] + '_id'
                assert getattr(vocab, idx_property) == vocab[v]

        # Test for serialize/deserailze from json
        json_str = vocab.to_json()
        new_vocab = Vocab.from_json(json_str)
        check_same_vocab(new_vocab, vocab)
        # Test for save/load from file
        while True:
            fname = '{}.json'.format(uuid.uuid4())
            if os.path.exists(fname):
                continue
            vocab.save(path=fname)
            new_vocab = Vocab.load(fname)
            check_same_vocab(new_vocab, vocab)
            os.remove(fname)
            break

    words = ['a', 'a', 'b', 'd', 'c', 'b', 'a', 'c', 'd', 'd', 'd']
    random.shuffle(words)
    counter = collections.Counter(words)
    vocab = Vocab(counter, max_size=2, min_freq=None)
    check_consistency(vocab)
    assert vocab.all_tokens == ['d', 'a', '<unk>']
    # Test for unknown token
    vocab = Vocab(tokens=counter, max_size=2, min_freq=None, unk_token='<unk2>')
    check_consistency(vocab)
    assert vocab.all_tokens == ['d', 'a', '<unk2>']

    vocab = Vocab(tokens=counter, max_size=None, min_freq=None,
                  pad_token=Vocab.PAD_TOKEN, eos_token=Vocab.EOS_TOKEN,
                  bos_token=Vocab.BOS_TOKEN, cls_token=Vocab.CLS_TOKEN,
                  sep_token=Vocab.SEP_TOKEN, mask_token=Vocab.MASK_TOKEN)
    check_consistency(vocab)
    assert vocab.unk_token == Vocab.UNK_TOKEN
    assert vocab.pad_token == Vocab.PAD_TOKEN
    assert vocab.eos_token == Vocab.EOS_TOKEN
    assert vocab.bos_token == Vocab.BOS_TOKEN
    assert vocab.cls_token == Vocab.CLS_TOKEN
    assert vocab.sep_token == Vocab.SEP_TOKEN
    assert vocab.mask_token == Vocab.MASK_TOKEN
    assert vocab.special_token_keys == ['unk_token', 'bos_token', 'cls_token', 'eos_token', 'mask_token', 'pad_token', 'sep_token']
    assert vocab.special_tokens == ['<unk>', '<bos>', '<cls>', '<eos>', '<mask>', '<pad>', '<sep>']
    assert vocab.all_tokens == ['d', 'a', 'c', 'b', '<unk>', '<bos>', '<cls>', '<eos>', '<mask>', '<pad>', '<sep>']

    vocab = Vocab(counter, bos_token=Vocab.BOS_TOKEN, eos_token=Vocab.EOS_TOKEN,
                  pad_token=Vocab.PAD_TOKEN)
    check_consistency(vocab)
    assert vocab.all_tokens == ['d', 'a', 'c', 'b', '<unk>', '<bos>', '<eos>', '<pad>']

    vocab = Vocab(counter, max_size=None, min_freq=None,
                  pad_token=Vocab.PAD_TOKEN, eos_token=Vocab.EOS_TOKEN,
                  bos_token=Vocab.BOS_TOKEN, mask_token='<mask2>',
                  other3_token='<other3>', other2_token='<other2>')
    check_consistency(vocab)
    assert vocab.all_tokens == ['d', 'a', 'c', 'b', '<unk>', '<bos>', '<eos>', '<mask2>', '<other2>', '<other3>', '<pad>']
    assert vocab.mask_token == '<mask2>'
    assert vocab.other2_token == '<other2>'
    assert vocab.other3_token == '<other3>'
    assert vocab.special_token_keys == ['unk_token', 'bos_token', 'eos_token', 'mask_token', 'other2_token', 'other3_token', 'pad_token']
    assert vocab.special_tokens == ['<unk>', '<bos>', '<eos>', '<mask2>', '<other2>', '<other3>', '<pad>']

    vocab = Vocab(counter, max_size=1, min_freq=10000, unk_token=None)
    check_consistency(vocab)
    assert vocab.all_tokens == []

    vocab = Vocab([], pad_token=Vocab.PAD_TOKEN, eos_token=Vocab.EOS_TOKEN,
                  bos_token=Vocab.BOS_TOKEN, mask_token='<mask2>')
    check_consistency(vocab)
    assert vocab.all_tokens == ['<unk>', '<bos>', '<eos>', '<mask2>', '<pad>']
    vocab = Vocab(pad_token=Vocab.PAD_TOKEN, eos_token=Vocab.EOS_TOKEN,
                  bos_token=Vocab.BOS_TOKEN, mask_token='<mask2>')
    check_consistency(vocab)
    assert vocab.all_tokens == ['<unk>', '<bos>', '<eos>', '<mask2>', '<pad>']

    vocab = Vocab(['<unk2>', '<pad>', '<bos>', '<eos>', '<mask>', 'a'],
                  pad_token=Vocab.PAD_TOKEN, eos_token=Vocab.EOS_TOKEN,
                  bos_token=Vocab.BOS_TOKEN, mask_token='<mask>')
    check_consistency(vocab)
    assert vocab.all_tokens == ['<unk2>', '<pad>', '<bos>', '<eos>', '<mask>', 'a', '<unk>']
    assert vocab.special_tokens == ['<pad>', '<bos>', '<eos>', '<mask>', '<unk>']
    assert vocab.special_token_keys == ['pad_token', 'bos_token', 'eos_token', 'mask_token', 'unk_token']

    # Check errors
    with pytest.raises(ValueError):
        vocab = Vocab(['a', 'a', 'a'])
    with pytest.raises(ValueError):
        vocab = Vocab(['a', 'b', 'c'], mask_token='<mask>', another_mask_token='<mask>')
    with pytest.raises(ValueError):
        vocab = Vocab(['a', 'b', 'c'], mask_token='<mask>', another_mask_token='<mask>')
    vocab = Vocab(['a', 'b', 'c'])
    check_consistency(vocab)
    
    # Check emoji
    all_tokens = ['<unk>', '😁']
    vocab = Vocab(all_tokens, unk_token='<unk>')
    vocab_file = str(uuid.uuid4()) + '.vocab'
    vocab.save(vocab_file)
    vocab = Vocab.load(vocab_file)
    assert vocab.all_tokens == all_tokens
    os.remove(vocab_file)