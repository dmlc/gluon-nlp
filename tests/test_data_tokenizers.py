import pytest
import random
import collections
import pickle
from uuid import uuid4
import os
import unicodedata
import tempfile
import gluonnlp
from gluonnlp.data.tokenizers import WhitespaceTokenizer, MosesTokenizer, JiebaTokenizer,\
    SpacyTokenizer, SubwordNMTTokenizer, YTTMTokenizer, SentencepieceTokenizer, \
    HuggingFaceBPETokenizer, HuggingFaceByteBPETokenizer, HuggingFaceWordPieceTokenizer, \
    HuggingFaceTokenizer
from gluonnlp.base import get_repo_url
from gluonnlp.data import Vocab
from gluonnlp.utils.misc import download


EN_SAMPLES = ['Four score and seven years ago our fathers brought forth on this continent, '
              'a new nation, conceived in Liberty, and dedicated to the proposition '
              'that all men are created equal.',
              'In spite of the debate going on for months about the photos of Özil with the '
              'Turkish President Recep Tayyip Erdogan, he regrets the return of '
              'the 92-match national player Özil.']
DE_SAMPLES = ['Goethe stammte aus einer angesehenen bürgerlichen Familie; sein Großvater'
              ' mütterlicherseits war als Stadtschultheiß höchster Justizbeamter der'
              ' Stadt Frankfurt, sein Vater Doktor der Rechte und kaiserlicher Rat.',
              '"Das ist eine Frage, die natürlich davon abhängt, dass man einmal ins '
              'Gespräch kommt, dass man mit ihm auch darüber spricht, warum er das eine '
              'oder andere offenbar so empfunden hat, wie das in seinem Statement niedergelegt'
              ' ist", sagte Grindel im Fußball-Podcast "Phrasenmäher" der "Bild-Zeitung.']
ZH_SAMPLES = ['苟活者在淡红的血色中，会依稀看见微茫的希望；真的猛士，将更奋然而前行。',
              '参加工作，哈尔滨工业大学无线电工程系电子仪器及测量技术专业毕业。']

SUBWORD_TEST_SAMPLES = ["Hello, y'all! How are you Ⅷ 😁 😁 😁 ?",
                        'GluonNLP is great！！！!!!',
                        "GluonNLP-Amazon-Haibin-Leonard-Sheng-Shuai-Xingjian...../:!@# 'abc'"]


def random_inject_space(sentence):
    words = sentence.split()
    ret = ''
    for i, word in enumerate(words):
        ret += word
        if i < len(words) - 1:
            n_space_tokens = random.randint(1, 10)
            for j in range(n_space_tokens):
                ret += random.choice([' ', '\t', '\r', '\n'])
    return ret


def verify_encode_token_with_offsets(tokenizer, all_sentences, gt_offsets=None):
    if gt_offsets is None:
        for sentences in [all_sentences[0], all_sentences]:
            enc_tokens = tokenizer.encode(sentences, str)
            tokens, offsets = tokenizer.encode_with_offsets(sentences, str)
            if isinstance(sentences, list):
                for ele_tokens, ele_enc_tokens, ele_offsets, ele_sentence in\
                        zip(tokens, enc_tokens, offsets, sentences):
                    for tok, offset, enc_tok in zip(ele_tokens, ele_offsets, ele_enc_tokens):
                        assert ele_sentence[offset[0]:offset[1]] == tok
                        assert tok == enc_tok
            else:
                for tok, offset, enc_tok in zip(tokens, offsets, enc_tokens):
                    assert sentences[offset[0]:offset[1]] == tok
                    assert tok == enc_tok
    else:
        for sentences, ele_gt_offsets in [(all_sentences[0], gt_offsets[0]),
                                          (all_sentences, gt_offsets)]:
            enc_tokens = tokenizer.encode(sentences, str)
            tokens, offsets = tokenizer.encode_with_offsets(sentences, str)
            assert ele_gt_offsets == offsets
            assert enc_tokens == tokens


def verify_sentencepiece_tokenizer_with_offsets(tokenizer, all_sentences):
    for sentences in [all_sentences[0], all_sentences]:
        enc_tokens = tokenizer.encode(sentences, str)
        tokens, offsets = tokenizer.encode_with_offsets(sentences, str)
        if isinstance(sentences, list):
            for ele_tokens, ele_enc_tokens, ele_offsets, ele_sentence\
                    in zip(tokens, enc_tokens, offsets, sentences):
                for i, (tok, offset, enc_tok) in enumerate(zip(ele_tokens, ele_offsets,
                                                               ele_enc_tokens)):
                    assert tok == enc_tok
                    ele_sel_tok = unicodedata.normalize('NFKC',
                                                        ele_sentence[offset[0]:offset[1]]).strip()
                    if tokenizer.is_first_subword(tok):
                        real_tok = tok[1:]
                    else:
                        real_tok = tok
                    assert ele_sel_tok == real_tok,\
                        'ele_sel_tok={}, real_tok={}'.format(ele_sel_tok, real_tok)


def verify_encode_with_offsets_consistency(tokenizer, all_sentences):
    for sentences in [all_sentences[0], all_sentences]:
        enc_tokens = tokenizer.encode(sentences, int)
        tokens, offsets = tokenizer.encode_with_offsets(sentences, int)
        str_tokens, str_offsets = tokenizer.encode_with_offsets(sentences, str)
        assert offsets == str_offsets
        assert tokens == enc_tokens


def verify_encode_token(tokenizer, all_sentences, all_gt_tokens):
    for sentences, gt_tokens in [(all_sentences[0], all_gt_tokens[0]),
                                 (all_sentences, all_gt_tokens)]:
        tokenizer_encode_ret = tokenizer.encode(sentences)
        assert tokenizer_encode_ret == gt_tokens,\
            'Whole Encoded: {}, \nWhole GT: {}'.format(tokenizer_encode_ret, gt_tokens)


def verify_decode(tokenizer, all_sentences, out_type=str):
    for sentences in [all_sentences[0], all_sentences]:
        assert tokenizer.decode(tokenizer.encode(sentences, out_type)) == sentences


def verify_decode_spm(tokenizer, all_sentences, gt_int_decode_sentences):
    for sentences, case_gt_int_decode in [(all_sentences[0], gt_int_decode_sentences[0]),
                                          (all_sentences, gt_int_decode_sentences)]:
        if isinstance(sentences, str):
            gt_str_decode_sentences = sentences
            if tokenizer.lowercase:
                gt_str_decode_sentences = gt_str_decode_sentences.lower()
            gt_str_decode_sentences = unicodedata.normalize('NFKC', gt_str_decode_sentences)
        elif isinstance(sentences, list):
            gt_str_decode_sentences = []
            for ele in sentences:
                ele_gt_decode = ele
                if tokenizer.lowercase:
                    ele_gt_decode = ele_gt_decode.lower()
                ele_gt_decode = unicodedata.normalize('NFKC', ele_gt_decode)
                gt_str_decode_sentences.append(ele_gt_decode)
        else:
            raise NotImplementedError
        assert tokenizer.decode(tokenizer.encode(sentences, str)) == gt_str_decode_sentences
        assert tokenizer.decode(tokenizer.encode(sentences, int)) == case_gt_int_decode


def verify_decode_subword_nmt(tokenizer, all_sentences, gt_int_decode, gt_str_decode):
    for sentences, case_gt_int_decode, case_gt_str_decode in [(all_sentences[0], gt_int_decode[0], gt_str_decode[0]),
                                                              (all_sentences, gt_int_decode, gt_str_decode)]:
        assert tokenizer.decode(tokenizer.encode(sentences, str)) == case_gt_str_decode
        assert tokenizer.decode(tokenizer.encode(sentences, int)) == case_gt_int_decode


def verify_decode_hf(tokenizer, all_sentences, gt_decode_sentences):
    for sentences, case_gt_decode in [(all_sentences[0], gt_decode_sentences[0]),
                                      (all_sentences, gt_decode_sentences)]:
        assert tokenizer.decode(tokenizer.encode(sentences, str)) == case_gt_decode
        assert tokenizer.decode(tokenizer.encode(sentences, int)) == case_gt_decode
        if isinstance(sentences, list):
            for sentence in sentences:
                assert tokenizer.vocab.to_tokens(tokenizer.encode(sentence, int))\
                       == tokenizer.encode(sentence, str)
                assert tokenizer.vocab[tokenizer.encode(sentence, str)]\
                       == tokenizer.encode(sentence, int)
        else:
            assert tokenizer.vocab.to_tokens(tokenizer.encode(sentences, int)) \
                   == tokenizer.encode(sentences, str)
            assert tokenizer.vocab[tokenizer.encode(sentences, str)] \
                   == tokenizer.encode(sentences, int)


def verify_decode_no_vocab_raise(tokenizer):
    # When the vocab is not attached, should raise ValueError
    for sentences in [EN_SAMPLES[0], EN_SAMPLES]:
        with pytest.raises(ValueError):
            tokenizer.encode(sentences, int)
    with pytest.raises(ValueError):
        tokenizer.decode([0])
    with pytest.raises(ValueError):
        tokenizer.decode([[0], [1]])


def verify_pickleble(tokenizer, cls):
    print(tokenizer)
    # Verify if the tokenizer is pickleable and has the same behavior after dumping/loading
    tokenizer_p = pickle.loads(pickle.dumps(tokenizer))
    assert isinstance(tokenizer_p, cls)
    assert tokenizer.encode(SUBWORD_TEST_SAMPLES, str) == tokenizer_p.encode(SUBWORD_TEST_SAMPLES, str)

def test_whitespace_tokenizer():
    tokenizer = WhitespaceTokenizer()
    gt_en_tokenized = [['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers', 'brought',
                        'forth', 'on', 'this', 'continent,', 'a', 'new', 'nation,', 'conceived',
                        'in', 'Liberty,', 'and', 'dedicated', 'to', 'the', 'proposition', 'that',
                        'all', 'men', 'are', 'created', 'equal.'],
                       ['In', 'spite', 'of', 'the', 'debate', 'going', 'on', 'for', 'months',
                        'about', 'the', 'photos', 'of', 'Özil', 'with', 'the', 'Turkish',
                        'President', 'Recep', 'Tayyip', 'Erdogan,', 'he', 'regrets', 'the',
                        'return', 'of', 'the', '92-match', 'national', 'player', 'Özil.']]
    gt_de_tokenized = [['Goethe', 'stammte', 'aus', 'einer', 'angesehenen', 'bürgerlichen',
                        'Familie;', 'sein', 'Großvater', 'mütterlicherseits', 'war', 'als',
                        'Stadtschultheiß', 'höchster', 'Justizbeamter', 'der', 'Stadt',
                        'Frankfurt,', 'sein', 'Vater', 'Doktor', 'der', 'Rechte', 'und',
                        'kaiserlicher', 'Rat.'],
                       ['"Das', 'ist', 'eine', 'Frage,', 'die', 'natürlich', 'davon', 'abhängt,',
                        'dass', 'man', 'einmal', 'ins', 'Gespräch', 'kommt,', 'dass', 'man', 'mit',
                        'ihm', 'auch', 'darüber', 'spricht,', 'warum', 'er', 'das', 'eine', 'oder',
                        'andere', 'offenbar', 'so', 'empfunden', 'hat,', 'wie', 'das', 'in',
                        'seinem', 'Statement', 'niedergelegt', 'ist",', 'sagte', 'Grindel', 'im',
                        'Fußball-Podcast', '"Phrasenmäher"', 'der', '"Bild-Zeitung.']]
    for _ in range(2):
        # Inject noise and test for encode
        noisy_en_samples = [random_inject_space(ele) for ele in EN_SAMPLES]
        noisy_de_samples = [random_inject_space(ele) for ele in DE_SAMPLES]
        verify_encode_token(tokenizer, noisy_en_samples + noisy_de_samples,
                            gt_en_tokenized + gt_de_tokenized)
        # Test for decode
        verify_decode(tokenizer, EN_SAMPLES + DE_SAMPLES, str)
        # Test for encode_with_offsets
        verify_encode_token_with_offsets(tokenizer, noisy_en_samples + noisy_de_samples)
    verify_decode_no_vocab_raise(tokenizer)

    # Test for output_type = int
    vocab = Vocab(collections.Counter(sum(gt_en_tokenized + gt_de_tokenized,
                                          [])))
    tokenizer.set_vocab(vocab)
    verify_decode(tokenizer, EN_SAMPLES + DE_SAMPLES, int)
    verify_pickleble(tokenizer, WhitespaceTokenizer)
    verify_encode_token_with_offsets(tokenizer, EN_SAMPLES + DE_SAMPLES)


def test_moses_tokenizer():
    en_tokenizer = MosesTokenizer('en')
    de_tokenizer = MosesTokenizer('de')
    gt_en_tokenized = [['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers',
                        'brought', 'forth', 'on', 'this', 'continent', ',', 'a', 'new', 'nation',
                        ',', 'conceived', 'in', 'Liberty', ',', 'and', 'dedicated', 'to', 'the',
                        'proposition', 'that', 'all', 'men', 'are', 'created', 'equal', '.'],
                       ['In', 'spite', 'of', 'the', 'debate', 'going', 'on', 'for', 'months',
                        'about', 'the', 'photos', 'of', 'Özil', 'with', 'the', 'Turkish',
                        'President', 'Recep', 'Tayyip', 'Erdogan', ',', 'he', 'regrets', 'the',
                        'return', 'of', 'the', '92-match', 'national', 'player', 'Özil', '.']]
    gt_de_tokenized = [['Goethe', 'stammte', 'aus', 'einer', 'angesehenen', 'bürgerlichen',
                        'Familie', ';', 'sein', 'Großvater', 'mütterlicherseits', 'war', 'als',
                        'Stadtschultheiß', 'höchster', 'Justizbeamter', 'der', 'Stadt',
                        'Frankfurt', ',', 'sein', 'Vater', 'Doktor', 'der', 'Rechte', 'und',
                        'kaiserlicher', 'Rat', '.'],
                       ['&quot;', 'Das', 'ist', 'eine', 'Frage', ',', 'die', 'natürlich', 'davon',
                        'abhängt', ',', 'dass', 'man', 'einmal', 'ins', 'Gespräch', 'kommt', ',',
                        'dass', 'man', 'mit', 'ihm', 'auch', 'darüber', 'spricht', ',', 'warum',
                        'er', 'das', 'eine', 'oder', 'andere', 'offenbar', 'so', 'empfunden',
                        'hat', ',', 'wie', 'das', 'in', 'seinem', 'Statement', 'niedergelegt',
                        'ist', '&quot;', ',', 'sagte', 'Grindel', 'im', 'Fußball-Podcast',
                        '&quot;', 'Phrasenmäher', '&quot;', 'der', '&quot;', 'Bild-Zeitung', '.']]
    verify_encode_token(en_tokenizer, EN_SAMPLES, gt_en_tokenized)
    verify_encode_token(de_tokenizer, DE_SAMPLES, gt_de_tokenized)
    verify_decode(en_tokenizer, EN_SAMPLES, str)
    verify_decode(de_tokenizer, DE_SAMPLES, str)
    vocab = Vocab(collections.Counter(sum(gt_en_tokenized + gt_de_tokenized, [])))
    verify_decode_no_vocab_raise(en_tokenizer)
    verify_decode_no_vocab_raise(de_tokenizer)
    en_tokenizer.set_vocab(vocab)
    de_tokenizer.set_vocab(vocab)
    verify_decode(en_tokenizer, EN_SAMPLES, int)
    verify_decode(de_tokenizer, DE_SAMPLES, int)
    verify_pickleble(en_tokenizer, MosesTokenizer)
    verify_pickleble(de_tokenizer, MosesTokenizer)


def test_jieba_tokenizer():
    tokenizer = JiebaTokenizer()
    gt_zh_tokenized = [['苟活', '者', '在', '淡红', '的', '血色', '中', '，',
                        '会', '依稀', '看见', '微茫', '的', '希望', '；', '真的',
                        '猛士', '，', '将', '更奋', '然而', '前行', '。'],
                       ['参加', '工作', '，', '哈尔滨工业大学', '无线电', '工程系', '电子仪器',
                        '及', '测量', '技术', '专业', '毕业', '。']]
    verify_encode_token(tokenizer, ZH_SAMPLES, gt_zh_tokenized)
    verify_decode(tokenizer, ZH_SAMPLES, str)
    vocab = Vocab(collections.Counter(sum(gt_zh_tokenized, [])))
    verify_decode_no_vocab_raise(tokenizer)
    tokenizer.set_vocab(vocab)
    verify_decode(tokenizer, ZH_SAMPLES, int)
    verify_pickleble(tokenizer, JiebaTokenizer)


def test_spacy_tokenizer():
    en_tokenizer = SpacyTokenizer('en')
    de_tokenizer = SpacyTokenizer('de')
    gt_en_tokenized = [['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers',
                        'brought', 'forth', 'on', 'this', 'continent', ',', 'a', 'new', 'nation',
                        ',', 'conceived', 'in', 'Liberty', ',', 'and', 'dedicated', 'to', 'the',
                        'proposition', 'that', 'all', 'men', 'are', 'created', 'equal', '.'],
                       ['In', 'spite', 'of', 'the', 'debate', 'going', 'on', 'for', 'months',
                        'about', 'the', 'photos', 'of', 'Özil', 'with', 'the', 'Turkish',
                        'President', 'Recep', 'Tayyip', 'Erdogan', ',', 'he', 'regrets', 'the',
                        'return', 'of', 'the', '92-match', 'national', 'player', 'Özil', '.']]
    gt_de_tokenized = [['Goethe', 'stammte', 'aus', 'einer', 'angesehenen', 'bürgerlichen',
                        'Familie', ';', 'sein', 'Großvater', 'mütterlicherseits', 'war', 'als',
                        'Stadtschultheiß', 'höchster', 'Justizbeamter', 'der', 'Stadt', 'Frankfurt',
                        ',', 'sein', 'Vater', 'Doktor', 'der', 'Rechte', 'und', 'kaiserlicher',
                        'Rat', '.'],
                       ['"', 'Das', 'ist', 'eine', 'Frage', ',', 'die', 'natürlich', 'davon',
                        'abhängt', ',', 'dass', 'man', 'einmal', 'ins', 'Gespräch', 'kommt', ',',
                        'dass', 'man', 'mit', 'ihm', 'auch', 'darüber', 'spricht', ',', 'warum',
                        'er', 'das', 'eine', 'oder', 'andere', 'offenbar', 'so', 'empfunden', 'hat',
                        ',', 'wie', 'das', 'in', 'seinem', 'Statement', 'niedergelegt', 'ist', '"',
                        ',', 'sagte', 'Grindel', 'im', 'Fußball-Podcast', '"', 'Phrasenmäher', '"',
                        'der', '"', 'Bild-Zeitung', '.']]
    verify_encode_token(en_tokenizer, EN_SAMPLES, gt_en_tokenized)
    verify_encode_token(de_tokenizer, DE_SAMPLES, gt_de_tokenized)
    vocab = Vocab(collections.Counter(sum(gt_en_tokenized + gt_de_tokenized, [])))
    en_tokenizer.set_vocab(vocab)
    de_tokenizer.set_vocab(vocab)
    verify_pickleble(en_tokenizer, SpacyTokenizer)
    verify_pickleble(de_tokenizer, SpacyTokenizer)
    verify_encode_token_with_offsets(en_tokenizer, EN_SAMPLES)
    verify_encode_token_with_offsets(de_tokenizer, DE_SAMPLES)

    # Test for loading spacy tokenizer from specifying the "model" flag
    en_tokenizer = SpacyTokenizer(model='en_core_web_lg')
    out = en_tokenizer.encode(EN_SAMPLES)


def test_yttm_tokenizer():
    with tempfile.TemporaryDirectory() as dir_path:
        model_path = os.path.join(dir_path, 'yttm.model')
        download(url=get_repo_url() + 'tokenizer_test_models/yttm/test_ende_yttm-6f2c39.model',
                 path=model_path)
        tokenizer = YTTMTokenizer(model_path=model_path)
        gt_tokenized = [['▁He', 'll', 'o', ',', '▁y', "'", 'all', '!', '▁How', '▁are', '▁you', '▁',
                         'Ⅷ', '▁', '😁', '▁', '😁', '▁', '😁', '▁?'],
                        ['▁Gl', 'u', 'on', 'N', 'L', 'P', '▁is', '▁great', '！', '！', '！', '!',
                         '!', '!'],
                        ['▁Gl', 'u', 'on', 'N', 'L', 'P', '-A', 'm', 'az', 'on', '-H', 'a', 'ib',
                         'in', '-L', 'e', 'on', 'ard', '-S', 'hen', 'g', '-S', 'h', 'u', 'ai',
                         '-', 'X', 'ing', 'j', 'ian', '.', '.', '.', '.', '.', '/', ':', '!',
                         '@', '#', '▁', "'", 'ab', 'c', "'"]]
        gt_offsets = [[(0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 9), (9, 12), (12, 13), (13, 17),
                       (17, 21), (21, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
                       (31, 32), (32, 33), (33, 35)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 11), (11, 17), (17, 18),
                       (18, 19), (19, 20), (20, 21), (21, 22), (22, 23)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 10), (10, 11), (11, 13),
                       (13, 15), (15, 17), (17, 18), (18, 20), (20, 22), (22, 24), (24, 25), (25, 27),
                       (27, 30), (30, 32), (32, 35), (35, 36), (36, 38), (38, 39), (39, 40), (40, 42),
                       (42, 43), (43, 44), (44, 47), (47, 48), (48, 51), (51, 52), (52, 53), (53, 54),
                       (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62),
                       (62, 63), (63, 65), (65, 66), (66, 67)]]
        gt_int_decode = ['Hello, y<UNK>all! How are you <UNK> <UNK> <UNK> <UNK> ?',
                         'GluonNLP is great！！！!!!',
                         'GluonNLP-Amazon-Haibin-Leonard-Sheng-Shuai-Xingjian...../:!@# <UNK>abc<UNK>']
        gt_str_decode = ["Hello, y'all! How are you Ⅷ 😁 😁 😁 ?",
                         'GluonNLP is great！！！!!!',
                         "GluonNLP-Amazon-Haibin-Leonard-Sheng-Shuai-Xingjian...../:!@# 'abc'"]
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, YTTMTokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        # Begin to verify decode
        for sample_sentences, ele_gt_int_decode, ele_gt_str_decode in [(SUBWORD_TEST_SAMPLES[0], gt_int_decode[0], gt_str_decode[0]),
                                                                       (SUBWORD_TEST_SAMPLES, gt_int_decode, gt_str_decode)]:
            int_decode = tokenizer.decode(tokenizer.encode(sample_sentences, int))
            str_decode = tokenizer.decode(tokenizer.encode(sample_sentences, str))
            assert int_decode == ele_gt_int_decode
            assert str_decode == ele_gt_str_decode
        os.remove(model_path)


@pytest.mark.seed(123)
def test_sentencepiece_tokenizer():
    with tempfile.TemporaryDirectory() as dir_path:
        model_path = os.path.join(dir_path, 'spm.model')
        download(url=get_repo_url()
                     + 'tokenizer_test_models/sentencepiece/case1/test_ende-a9bee4.model',
                 path=model_path)
        # Case1
        tokenizer = SentencepieceTokenizer(model_path)
        gt_tokenized = [['▁Hel', 'lo', ',', '▁y', "'", 'all', '!', '▁How', '▁are', '▁you',
                         '▁', 'VI', 'II', '▁', '😁', '▁', '😁', '▁', '😁', '▁?'],
                        ['▁G', 'lu', 'on', 'N', 'L', 'P', '▁is', '▁great', '!', '!', '!', '!',
                         '!', '!'],
                        ['▁G', 'lu', 'on', 'N', 'L', 'P', '-', 'A', 'ma', 'zo', 'n', '-', 'H', 'ai',
                         'bin', '-', 'L', 'e', 'on', 'ard', '-', 'S', 'hen', 'g', '-', 'S', 'hu', 'ai',
                         '-', 'X', 'ing', 'j', 'ian', '.', '.', '.', '.', '.', '/', ':', '!', '@',
                         '#', '▁', "'", 'ab', 'c', "'"]]
        gt_offsets = [[(0, 3), (3, 5), (5, 6), (6, 8), (8, 9), (9, 12), (12, 13), (13, 17), (17, 21),
                       (21, 25), (25, 26), (26, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
                       (31, 32), (32, 33), (33, 35)],
                      [(0, 1), (1, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 11), (11, 17), (17, 18),
                       (18, 19), (19, 20), (20, 21), (21, 22), (22, 23)],
                      [(0, 1), (1, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 12),
                       (12, 14), (14, 15), (15, 16), (16, 17), (17, 19), (19, 22), (22, 23), (23, 24),
                       (24, 25), (25, 27), (27, 30), (30, 31), (31, 32), (32, 35), (35, 36), (36, 37),
                       (37, 38), (38, 40), (40, 42), (42, 43), (43, 44), (44, 47), (47, 48), (48, 51),
                       (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59),
                       (59, 60), (60, 61), (61, 62), (62, 63), (63, 65), (65, 66), (66, 67)]]
        gt_int_decode = ['Hello, y ⁇ all! How are you VIII  ⁇   ⁇   ⁇  ?',
                         'GluonNLP is great!!!!!!',
                         'GluonNLP-Amazon-Haibin-Leonard-Sheng-Shuai-Xingjian...../:! ⁇ #  ⁇ abc ⁇ ']
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, SentencepieceTokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_spm(tokenizer, SUBWORD_TEST_SAMPLES, gt_int_decode)

        # Case2, lower_case
        gt_lower_case_int_decode = ['hello, y ⁇ all! how are you viii  ⁇   ⁇   ⁇  ?',
                                    'gluonnlp is great!!!!!!',
                                    'gluonnlp-amazon-haibin-leonard-sheng-shuai-xingjian...../:! ⁇ #  ⁇ abc ⁇ ']
        tokenizer = SentencepieceTokenizer(model_path, lowercase=True)
        verify_decode_spm(tokenizer, SUBWORD_TEST_SAMPLES, gt_lower_case_int_decode)

        # Case3, Use the sentencepiece regularization commands, we test whether we can obtain different encoding results
        tokenizer = SentencepieceTokenizer(model_path, lowercase=True, nbest=-1, alpha=1.0)
        has_different_encode_out = False
        encode_out = None
        for _ in range(10):
            if encode_out is None:
                encode_out = tokenizer.encode(SUBWORD_TEST_SAMPLES[0])
            else:
                ele_out = tokenizer.encode(SUBWORD_TEST_SAMPLES[0])
                if ele_out != encode_out:
                    has_different_encode_out = True
                    break
        assert has_different_encode_out
        os.remove(model_path)


def test_subword_nmt_tokenizer():
    with tempfile.TemporaryDirectory() as dir_path:
        model_path = os.path.join(dir_path, 'subword_nmt.model')
        download(url=get_repo_url() + 'tokenizer_test_models/subword-nmt/test_ende-d189ff.model',
                 path=model_path)
        vocab_path = os.path.join(dir_path, 'subword_nmt.vocab')
        download(url=get_repo_url() + 'tokenizer_test_models/subword-nmt/test_ende_vocab-900f81.json',
                 path=vocab_path)

        # Case 1
        tokenizer = SubwordNMTTokenizer(model_path, vocab_path)
        gt_tokenized = [["Hel", "lo", ",</w>", "y", "\'", "all", "!</w>", "How</w>", "are</w>", "you</w>",
                         "Ⅷ</w>", "😁</w>", "😁</w>", "😁</w>", "?</w>"],
                        ["Gl", "u", "on", "N", "L", "P</w>", "is</w>", "great", "！", "！", "！", "!!",
                         "!</w>"],
                        ["Gl", "u", "on", "N", "L", "P", "-", "Amaz", "on-", "H", "ai", "b", "in-", "Le",
                         "on", "ard", "-", "Sh", "eng", "-", "Sh", "u", "ai", "-", "X", "ing", "ji",
                         "an", "..", "...", "/", ":", "!", "@", "#</w>", "\'", "ab", "c", "\'</w>"]]
        gt_offsets = [[(0, 3), (3, 5), (5, 6), (7, 8), (8, 9), (9, 12), (12, 13), (14, 17), (18, 21),
                       (22, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (9, 11), (12, 17), (17, 18),
                       (18, 19), (19, 20), (20, 22), (22, 23)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 13), (13, 16),
                       (16, 17), (17, 19), (19, 20), (20, 23), (23, 25), (25, 27), (27, 30), (30, 31),
                       (31, 33), (33, 36), (36, 37), (37, 39), (39, 40), (40, 42), (42, 43), (43, 44),
                       (44, 47), (47, 49), (49, 51), (51, 53), (53, 56), (56, 57), (57, 58), (58, 59),
                       (59, 60), (60, 61), (62, 63), (63, 65), (65, 66), (66, 67)]]
        gt_int_decode = ["Hello, y\'all! How are you Ⅷ 😁 😁 😁 ?",
                         "GluonNLP is great！！！!!!",
                         "GluonNLP-Amazon-Haibin-Leonard-Sheng-Shuai-Xingjian...../:!@# \'abc\'"]
        gt_str_decode = SUBWORD_TEST_SAMPLES
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, SubwordNMTTokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_subword_nmt(tokenizer, SUBWORD_TEST_SAMPLES, gt_int_decode, gt_str_decode)

        # Case 2, bpe_dropout
        # We use str decode here because we may not perfectly recover the original sentence with int decode.
        tokenizer = SubwordNMTTokenizer(model_path, vocab_path, bpe_dropout=0.5)
        verify_decode(tokenizer, SUBWORD_TEST_SAMPLES, out_type=str)

        os.remove(model_path)
        os.remove(vocab_path)


def test_huggingface_bpe_tokenizer():
    with tempfile.TemporaryDirectory() as dir_path:
        model_path = os.path.join(dir_path, 'test_hf_bpe.model')
        download(url=get_repo_url() + 'tokenizer_test_models/hf_bpe/test_hf_bpe.model',
                 path=model_path)
        vocab_path = os.path.join(dir_path, 'test_hf_bpe.vocab')
        download(url=get_repo_url() + 'tokenizer_test_models/hf_bpe/test_hf_bpe.vocab',
                 path=vocab_path)
        hf_vocab_path = os.path.join(dir_path, 'test_hf_bpe.hf_vocab')
        download(url=get_repo_url() + 'tokenizer_test_models/hf_bpe/test_hf_bpe.hf_vocab',
                 path=hf_vocab_path)

        # Case 1, default lowercase=False
        tokenizer = HuggingFaceBPETokenizer(model_path, vocab_path)
        gt_tokenized = [['Hello</w>', ',</w>', 'y</w>', "'</w>", 'all</w>', '!</w>', 'How</w>',
                         'are</w>', 'you</w>', '<unk>', '<unk>', '<unk>', '<unk>', '?</w>'],
                        ['Gl', 'u', 'on', 'N', 'LP</w>', 'is</w>', 'great</w>', '！</w>', '！</w>',
                         '！</w>', '!</w>', '!</w>', '!</w>'],
                        ['Gl', 'u', 'on', 'N', 'LP</w>', '-</w>', 'Amazon</w>', '-</w>', 'H', 'ai',
                         'bin</w>', '-</w>', 'Leonard</w>', '-</w>', 'Sh', 'en', 'g</w>', '-</w>',
                         'Sh', 'u', 'ai</w>', '-</w>', 'X', 'ing', 'j', 'ian</w>', '.</w>', '.</w>',
                         '.</w>', '.</w>', '.</w>', '/</w>', ':</w>', '!</w>', '@</w>', '#</w>',
                         "'</w>", 'ab', 'c</w>', "'</w>"]]
        gt_offsets = [[(0, 5), (5, 6), (7, 8), (8, 9), (9, 12), (12, 13), (14, 17), (18, 21), (22, 25),
                       (26, 27), (28, 29), (30, 31), (32, 33), (34, 35)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 8), (9, 11), (12, 17), (17, 18), (18, 19),
                       (19, 20), (20, 21), (21, 22), (22, 23)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 8), (8, 9), (9, 15), (15, 16), (16, 17),
                       (17, 19), (19, 22), (22, 23), (23, 30), (30, 31), (31, 33), (33, 35), (35, 36),
                       (36, 37), (37, 39), (39, 40), (40, 42), (42, 43), (43, 44), (44, 47), (47, 48),
                       (48, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58),
                       (58, 59), (59, 60), (60, 61), (62, 63), (63, 65), (65, 66), (66, 67)]]
        # gt_int_decode = gt_str_decode for hf
        # hf removed the unk tokens in decode result
        gt_decode = ["Hello , y ' all ! How are you ?",
                     'GluonNLP is great ！ ！ ！ ! ! !',
                     "GluonNLP - Amazon - Haibin - Leonard - Sheng - Shuai - Xingjian . . . . . / : ! @ # ' abc '"]
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceBPETokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)

        # Case 2, lowercase=True
        gt_lowercase_decode = ["hello , y ' all ! how are you ?",
                               'gluonnlp is great ！ ！ ！ ! ! !',
                               "gluonnlp - amazon - haibin - leonard - sheng - shuai - xingjian . . . . . / : ! @ # ' abc '"]
        tokenizer = HuggingFaceBPETokenizer(model_path, vocab_path, lowercase=True)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_lowercase_decode)

        # Case 3, using original hf vocab
        tokenizer = HuggingFaceBPETokenizer(model_path, hf_vocab_path)
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceBPETokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)

        os.remove(model_path)
        os.remove(vocab_path)
        os.remove(hf_vocab_path)


def test_huggingface_bytebpe_tokenizer():
    with tempfile.TemporaryDirectory() as dir_path:
        model_path = os.path.join(dir_path, 'hf_bytebpe.model')
        download(url=get_repo_url() + 'tokenizer_test_models/hf_bytebpe/test_hf_bytebpe.model',
                 path=model_path)
        vocab_path = os.path.join(dir_path, 'hf_bytebpe.vocab')
        download(url=get_repo_url() + 'tokenizer_test_models/hf_bytebpe/test_hf_bytebpe.vocab',
                 path=vocab_path)
        hf_vocab_path = os.path.join(dir_path, 'hf_bytebpe.hf_vocab')
        download(url=get_repo_url() + 'tokenizer_test_models/hf_bytebpe/test_hf_bytebpe.hf_vocab',
                 path=hf_vocab_path)

        # Case 1, default lowercase=False
        tokenizer = HuggingFaceByteBPETokenizer(model_path, vocab_path)
        gt_tokenized = [['Hello', ',', 'Ġy', "'", 'all', '!', 'ĠHow', 'Ġare', 'Ġyou',
                         'Ġâ', 'ħ', '§', 'ĠðŁĺ', 'ģ', 'ĠðŁĺ', 'ģ', 'ĠðŁĺ', 'ģ', 'Ġ?'],
                        ['Gl', 'u', 'on', 'N', 'LP', 'Ġis', 'Ġgreat', 'ï¼', 'ģ', 'ï¼',
                         'ģ', 'ï¼', 'ģ', '!!!'],
                        ['Gl', 'u', 'on', 'N', 'LP', '-', 'Amazon', '-', 'Ha', 'ib', 'in',
                         '-', 'Le', 'on', 'ard', '-', 'She', 'ng', '-', 'Sh', 'u',
                         'ai', '-', 'X', 'ing', 'j', 'ian', '.....', '/', ':', '!', '@',
                         '#', "Ġ'", 'ab', 'c', "'"]]
        # the defination of the offsets of bytelevel seems not clear
        gt_offsets = [[(0, 5), (5, 6), (6, 8), (8, 9), (9, 12), (12, 13), (13, 17), (17, 21),
                       (21, 25), (25, 27), (26, 27), (26, 27), (27, 29), (28, 29), (29, 31),
                       (30, 31), (31, 33), (32, 33), (33, 35)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 8), (8, 11), (11, 17), (17, 18),
                       (17, 18), (18, 19), (18, 19), (19, 20), (19, 20), (20, 23)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 8), (8, 9), (9, 15), (15, 16),
                       (16, 18), (18, 20), (20, 22), (22, 23), (23, 25), (25, 27), (27, 30),
                       (30, 31), (31, 34), (34, 36), (36, 37), (37, 39), (39, 40), (40, 42),
                       (42, 43), (43, 44), (44, 47), (47, 48), (48, 51), (51, 56),
                       (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 63),
                       (63, 65), (65, 66), (66, 67)]]
        gt_decode = ["Hello, y'all! How are you Ⅷ 😁 😁 😁 ?",
                     'GluonNLP is great！！！!!!',
                     "GluonNLP-Amazon-Haibin-Leonard-Sheng-Shuai-Xingjian...../:!@# 'abc'"]
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceByteBPETokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)

        # Case 2, lowercase=True
        gt_lowercase_int_decode = ["hello, y'all! how are you ⅷ 😁 😁 😁 ?",
                                   'gluonnlp is great！！！!!!',
                                   "gluonnlp-amazon-haibin-leonard-sheng-shuai-xingjian...../:!@# 'abc'"]
        tokenizer = HuggingFaceByteBPETokenizer(model_path, vocab_path, lowercase=True)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_lowercase_int_decode)

        # Case 3, using original hf vocab
        tokenizer = HuggingFaceByteBPETokenizer(model_path, hf_vocab_path)
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceByteBPETokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)

        os.remove(model_path)
        os.remove(vocab_path)
        os.remove(hf_vocab_path)


def test_huggingface_wordpiece_tokenizer():
    with tempfile.TemporaryDirectory() as dir_path:
        vocab_path = os.path.join(dir_path, 'hf_wordpiece.vocab')
        download(url=get_repo_url()
                     + 'tokenizer_test_models/hf_wordpiece/test_hf_wordpiece.vocab',
                 path=vocab_path)
        hf_vocab_path = os.path.join(dir_path, 'hf_wordpiece.hf_vocab')
        download(url=get_repo_url()
                     + 'tokenizer_test_models/hf_wordpiece/test_hf_wordpiece.hf_vocab',
                 path=hf_vocab_path)

        # Case 1, lowercase=True
        tokenizer = HuggingFaceWordPieceTokenizer(vocab_path, lowercase=True)
        gt_tokenized = [["hello", ",", "y", "'", "all", "!", "how", "are", "you",
                         "<unk>", "<unk>", "<unk>", "<unk>", "?"],
                        ["gl", "##uo", "##nn", "##l", "##p", "is", "great", "\uff01",
                         "\uff01", "\uff01", "!", "!", "!"],
                        ["gl", "##uo", "##nn", "##l", "##p", "-", "amazon", "-", "hai",
                         "##bin", "-", "leonard", "-", "shen", "##g", "-", "shu", "##ai", "-",
                         "xin", "##g", "##ji", "##an", ".", ".", ".", ".", ".", "/", ":", "!",
                         "@", "#", "'", "abc", "'"]]
        gt_offsets = [[(0, 5), (5, 6), (7, 8), (8, 9), (9, 12), (12, 13), (14, 17), (18, 21),
                       (22, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35)],
                      [(0, 2), (2, 4), (4, 6), (6, 7), (7, 8), (9, 11), (12, 17), (17, 18),
                       (18, 19), (19, 20), (20, 21), (21, 22), (22, 23)],
                      [(0, 2), (2, 4), (4, 6), (6, 7), (7, 8), (8, 9), (9, 15), (15, 16), (16, 19),
                       (19, 22), (22, 23), (23, 30), (30, 31), (31, 35), (35, 36), (36, 37), (37, 40),
                       (40, 42), (42, 43), (43, 46), (46, 47), (47, 49), (49, 51), (51, 52), (52, 53),
                       (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61),
                       (62, 63), (63, 66), (66, 67)]]
        gt_decode = ["hello, y'all! how are you?",
                     "gluonnlp is great ！ ！ ！!!!",
                     "gluonnlp - amazon - haibin - leonard - sheng - shuai - xingjian..... / :! @ #'abc '"]
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceWordPieceTokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)

        # Case 2, lowercase=False
        gt_lowercase_decode = [", y'all! are you?",
                               "is great ！ ！ ！!!!",
                               "- - - - - -..... / :! @ #'abc '"]
        tokenizer = HuggingFaceWordPieceTokenizer(vocab_path, lowercase=False)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_lowercase_decode)

        # Case 3, using original hf vocab
        tokenizer = HuggingFaceWordPieceTokenizer(hf_vocab_path, lowercase=True)
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceWordPieceTokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)

        os.remove(vocab_path)
        os.remove(hf_vocab_path)


def test_huggingface_wordpiece_tokenizer_v08():
    """Test for huggingface tokenizer >=0.8"""
    with tempfile.TemporaryDirectory() as dir_path:
        model_path = os.path.join(dir_path, 'hf_wordpiece_new_0.8.model')
        download(url=get_repo_url() +
                     'tokenizer_test_models/hf_wordpiece_new_0.8/hf_wordpiece.model',
                 path=model_path,
                 sha1_hash='66ccadf6e5e354ff9604e4a82f107a2ac873abd5')
        vocab_path = os.path.join(dir_path, 'hf_wordpiece_new_0.8.vocab')
        download(url=get_repo_url() +
                     'tokenizer_test_models/hf_wordpiece_new_0.8/hf_wordpiece.vocab',
                 path=vocab_path,
                 sha1_hash='dd6fdf4bbc74eaa8806d12cb3d38a4d9a306aea8')
        tokenizer = HuggingFaceTokenizer(model_path, vocab_path)
        gt_tokenized = [['Hel', '##lo', ',', 'y', '[UNK]', 'all', '!',
                         'How', 'are', 'you', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '?'],
                        ['Gl', '##u', '##on', '##N', '##L', '##P', 'is', 'great', '[UNK]',
                         '[UNK]', '[UNK]', '!', '!', '!'],
                        ['Gl', '##u', '##on', '##N', '##L', '##P', '-',
                         'Am', '##az', '##on', '-', 'Ha', '##ibi', '##n', '-', 'Leon', '##ard',
                         '-', 'She', '##n', '##g', '-', 'Sh', '##ua', '##i', '-', 'X',
                         '##ing', '##j', '##ian', '.', '.', '.', '.', '.', '/', ':', '!',
                         '@', '#', '[UNK]', 'ab', '##c', '[UNK]']]
        gt_offsets = [[(0, 3), (3, 5), (5, 6), (7, 8), (8, 9), (9, 12), (12, 13),
                       (14, 17), (18, 21), (22, 25), (26, 27), (28, 29), (30, 31),
                       (32, 33), (34, 35)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (9, 11), (12, 17),
                       (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23)],
                      [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                       (9, 11), (11, 13), (13, 15), (15, 16), (16, 18), (18, 21),
                       (21, 22), (22, 23), (23, 27), (27, 30), (30, 31), (31, 34),
                       (34, 35), (35, 36), (36, 37), (37, 39), (39, 41), (41, 42),
                       (42, 43), (43, 44), (44, 47), (47, 48), (48, 51), (51, 52),
                       (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58),
                       (58, 59), (59, 60), (60, 61), (62, 63), (63, 65), (65, 66),
                       (66, 67)]]
        gt_decode = ['Hello, y all! How are you?',
                     'GluonNLP is great!!!',
                     'GluonNLP - Amazon - Haibin - Leonard - Sheng - Shuai - Xingjian..... / '
                     ':! @ # abc']
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceTokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)


def test_huggingface_bpe_tokenizer_v08():
    """Test for huggingface BPE tokenizer >=0.8"""
    with tempfile.TemporaryDirectory() as dir_path:
        model_path = os.path.join(dir_path, 'hf_bpe_new_0.8.model')
        download(url=get_repo_url() +
                     'tokenizer_test_models/hf_bpe_new_0.8/hf_bpe.model',
                 path=model_path,
                 sha1_hash='ecda90979561ca4c5a8d769b5e3c9fa2270d5317')
        vocab_path = os.path.join(dir_path, 'hf_bpe_new_0.8.vocab')
        download(url=get_repo_url() +
                     'tokenizer_test_models/hf_bpe_new_0.8/hf_bpe.vocab',
                 path=vocab_path,
                 sha1_hash='b92dde0b094f405208f3ec94b5eae88430bf4262')
        tokenizer = HuggingFaceTokenizer(model_path, vocab_path)
        gt_tokenized = [['H', 'ello</w>', ',</w>', 'y</w>', 'all</w>', '!</w>',
                         'How</w>', 'are</w>', 'you</w>', '?</w>'],
                        ['G', 'lu', 'on', 'N', 'L', 'P</w>', 'is</w>', 'great</w>',
                         '!</w>', '!</w>', '!</w>'],
                        ['G', 'lu', 'on', 'N', 'L', 'P</w>', '-</w>', 'Amaz', 'on</w>',
                         '-</w>', 'Ha', 'i', 'bin</w>', '-</w>', 'Leon', 'ard</w>', '-</w>',
                         'Sh', 'eng</w>', '-</w>', 'S', 'hu', 'ai</w>', '-</w>', 'X', 'ing',
                         'j', 'ian</w>', '.</w>', '.</w>', '.</w>', '.</w>', '.</w>', '/</w>',
                         ':</w>', '!</w>', '@</w>', '#</w>', 'ab', 'c</w>']]
        gt_offsets = [[(0, 1), (1, 5), (5, 6), (7, 8), (9, 12), (12, 13), (14, 17),
                       (18, 21), (22, 25), (34, 35)],
                      [(0, 1), (1, 3), (3, 5), (5, 6), (6, 7), (7, 8), (9, 11), (12, 17),
                       (20, 21), (21, 22), (22, 23)],
                      [(0, 1), (1, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 13), (13, 15),
                       (15, 16), (16, 18), (18, 19), (19, 22), (22, 23), (23, 27), (27, 30),
                       (30, 31), (31, 33), (33, 36), (36, 37), (37, 38), (38, 40), (40, 42),
                       (42, 43), (43, 44), (44, 47), (47, 48), (48, 51), (51, 52), (52, 53),
                       (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60),
                       (60, 61), (63, 65), (65, 66)]]
        gt_decode = ['Hello , y all ! How are you ?',
                     'GluonNLP is great ! ! !',
                     'GluonNLP - Amazon - Haibin - Leonard - Sheng - Shuai - Xingjian'
                     ' . . . . . / : ! @ # abc']
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceTokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)


def test_huggingface_bytebpe_tokenizer_v08():
    """Test for huggingface bytebpe tokenizer >=0.8"""
    with tempfile.TemporaryDirectory() as dir_path:
        model_path = os.path.join(dir_path, 'hf_bytebpe_new_0.8.model')
        download(url=get_repo_url() +
                     'tokenizer_test_models/hf_bytebpe_new_0.8/hf_bytebpe.model',
                 path=model_path,
                 sha1_hash='a1c4da1f6c21df923e150f56dbb5b7a53c61808b')
        vocab_path = os.path.join(dir_path, 'hf_bytebpe_new_0.8.vocab')
        download(url=get_repo_url() +
                     'tokenizer_test_models/hf_bytebpe_new_0.8/hf_bytebpe.vocab',
                 path=vocab_path,
                 sha1_hash='7831b19078a3222f450e65b2188dc0770473123b')
        tokenizer = HuggingFaceTokenizer(model_path, vocab_path)
        gt_tokenized = [['He', 'llo', ',', 'Ġy', "'", 'all', '!', 'ĠHow', 'Ġare', 'Ġyou',
                         'Ġâ', 'ħ', '§', 'Ġ', 'ð', 'Ł', 'ĺ', 'ģ', 'Ġ', 'ð', 'Ł', 'ĺ',
                         'ģ', 'Ġ', 'ð', 'Ł', 'ĺ', 'ģ', 'Ġ?'],
                        ['G', 'l', 'u', 'on', 'N', 'L', 'P', 'Ġis', 'Ġgreat', 'ï', '¼', 'ģ',
                         'ï', '¼', 'ģ', 'ï', '¼', 'ģ', '!', '!', '!'],
                        ['G', 'l', 'u', 'on', 'N', 'L', 'P', '-', 'Am', 'az', 'on', '-',
                         'Ha', 'ib', 'in', '-', 'Le', 'on', 'ard', '-', 'S', 'hen', 'g', '-',
                         'Sh', 'u', 'ai', '-', 'X', 'ing', 'j', 'ian',
                         '..', '...', '/', ':', '!', '@', '#', 'Ġ', "'", 'ab', 'c', "'"]]
        gt_offsets = [[(0, 2), (2, 5), (5, 6), (6, 8), (8, 9), (9, 12), (12, 13), (13, 17),
                       (17, 21), (21, 25), (25, 27), (26, 27), (26, 27), (27, 28), (28, 29),
                       (28, 29), (28, 29), (28, 29), (29, 30), (30, 31), (30, 31), (30, 31),
                       (30, 31), (31, 32), (32, 33), (32, 33), (32, 33), (32, 33), (33, 35)],
                      [(0, 1), (1, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 11), (11, 17),
                       (17, 18), (17, 18), (17, 18), (18, 19), (18, 19), (18, 19), (19, 20),
                       (19, 20), (19, 20), (20, 21), (21, 22), (22, 23)],
                      [(0, 1), (1, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 11),
                       (11, 13), (13, 15), (15, 16), (16, 18), (18, 20), (20, 22), (22, 23),
                       (23, 25), (25, 27), (27, 30), (30, 31), (31, 32), (32, 35), (35, 36),
                       (36, 37), (37, 39), (39, 40), (40, 42), (42, 43), (43, 44),
                       (44, 47), (47, 48), (48, 51), (51, 53), (53, 56), (56, 57),
                       (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63),
                       (63, 65), (65, 66), (66, 67)]]
        gt_decode = ["Hello, y'all! How are you Ⅷ 😁 😁 😁 ?",
                     'GluonNLP is great！！！!!!',
                     "GluonNLP-Amazon-Haibin-Leonard-Sheng-Shuai-Xingjian...../:!@# 'abc'"]
        verify_encode_token(tokenizer, SUBWORD_TEST_SAMPLES, gt_tokenized)
        verify_pickleble(tokenizer, HuggingFaceTokenizer)
        verify_encode_token_with_offsets(tokenizer, SUBWORD_TEST_SAMPLES, gt_offsets)
        verify_decode_hf(tokenizer, SUBWORD_TEST_SAMPLES, gt_decode)


def test_tokenizers_create():
    tokenizer = gluonnlp.data.tokenizers.create('moses', 'en')
    tokenizer.encode('hello world!')
