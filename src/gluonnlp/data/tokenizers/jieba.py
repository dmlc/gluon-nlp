__all__ = ['JiebaTokenizer']

from typing import Optional
from .base import *
from ..vocab import Vocab
from ...utils.lazy_imports import try_import_jieba


@TOKENIZER_REGISTRY.register('jieba')
class JiebaTokenizer(BaseTokenizerWithVocab):
    r"""Apply the jieba tokenizer to tokenize Chinese sentences.

    For more details, you may refer to [jieba](https://github.com/fxsjy/jieba)

    """

    def __init__(self, dictionary=None, vocab: Optional[Vocab] = None):
        self._vocab = vocab
        jieba = try_import_jieba()
        self._tokenizer = jieba.Tokenizer(dictionary)
        self._tokenizer.initialize(self._tokenizer.dictionary)

    def encode(self, sentences, output_type=str):
        if output_type is str:
            if isinstance(sentences, list):
                return [list(self._tokenizer.cut(sentence)) for sentence in sentences]
            else:
                return list(self._tokenizer.cut(sentences))
        elif output_type is int or output_type == 'id':
            if self._vocab is None:
                raise TokenizerEncodeWithoutVocabError
            if isinstance(sentences, list):
                return [[self._vocab[ele] for ele in self._tokenizer.cut(sentence)]
                        for sentence in sentences]
            else:
                return [self._vocab[ele] for ele in self._tokenizer.cut(sentences)]
        else:
            raise TokenTypeNotSupportedError(output_type)

    def encode_with_offsets(self, sentences, output_type=str):
        is_multiple_sentences = isinstance(sentences, list)
        if not is_multiple_sentences:
            sentences = [sentences]
        all_tokens = [list(self._tokenizer.tokenize(sentence)) for sentence in sentences]
        offsests = [[(ele[1], ele[2]) for ele in tokens] for tokens in all_tokens]
        if output_type is str:
            ret_tokens = [[ele[0] for ele in tokens] for tokens in all_tokens]
        elif output_type is int:
            if self._vocab is None:
                raise TokenizerEncodeWithoutVocabError
            ret_tokens = [self._vocab[[ele[0] for ele in tokens]] for tokens in all_tokens]
        else:
            raise TokenTypeNotSupportedError(output_type)
        if is_multiple_sentences:
            return ret_tokens, offsests
        else:
            return ret_tokens[0], offsests[0]

    def decode(self, tokens):
        is_multiple_sentences = is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = get_token_type(tokens)
        if token_type is str:
            ret = [''.join(ele_tokens) for ele_tokens in tokens]
        elif token_type is int:
            if self._vocab is None:
                raise TokenizerDecodeWithoutVocabError
            ret = [''.join(self._vocab.to_tokens(ele_tokens)) for ele_tokens in tokens]
        else:
            raise TokenTypeNotSupportedError(token_type)
        if is_multiple_sentences:
            return ret
        else:
            return ret[0]

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        self._vocab = vocab

    def __getstate__(self):
        """Make the JiebaTokenizer pickleble. It is safe to remove the lock."""
        d = {k: v for k, v in self._tokenizer.__dict__.items() if k != 'lock'}
        return d

    def __setstate__(self, state):
        jieba = try_import_jieba()
        self._tokenizer = jieba.Tokenizer()
        for k, v in state.items():
            setattr(self._tokenizer, k, v)

