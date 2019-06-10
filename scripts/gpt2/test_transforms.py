# coding: utf-8
import io
from transforms import GPT2Tokenizer, GPT2Detokenizer
from gluonnlp.vocab import Vocab

def test_gpt2_transformer():
    tokenizer = GPT2Tokenizer('models/345M/bpe_ranks.json')
    detokenizer = GPT2Detokenizer(tokenizer)
    with io.open('models/345M/vocab.json', 'r', encoding='utf-8') as f:
        vocab = Vocab.from_json(f.read())
    s = ' natural language processing tools such as gluonnlp and torchtext'
    subwords = tokenizer(s)
    indices = vocab[subwords]
    gt_gpt2_subword = ['Ġnatural', 'Ġlanguage', 'Ġprocessing', 'Ġtools', 'Ġsuch', 'Ġas', 'Ġgl', 'u', 'on',
                       'nl', 'p', 'Ġand', 'Ġtorch', 'text']
    gt_gpt2_idx = [3288, 3303, 7587, 4899, 884, 355, 1278, 84, 261, 21283, 79, 290, 28034, 5239]
    for lhs, rhs in zip(subwords, gt_gpt2_subword):
        assert lhs == rhs
    for lhs, rhs in zip(indices, gt_gpt2_idx):
        assert lhs == rhs

    recovered_sentence = detokenizer([vocab.idx_to_token[i] for i in indices])
    assert recovered_sentence == s
