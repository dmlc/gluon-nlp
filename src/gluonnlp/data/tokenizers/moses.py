__all__ = ['MosesTokenizer']

from typing import Optional
import warnings
import sacremoses
from .base import TOKENIZER_REGISTRY, BaseTokenizerWithVocab, \
    TokenizerEncodeWithoutVocabError, TokenizerDecodeWithoutVocabError,\
    TokenTypeNotSupportedError, is_tokens_from_multiple_sentences, get_token_type
from ..vocab import Vocab


@TOKENIZER_REGISTRY.register('moses')
class MosesTokenizer(BaseTokenizerWithVocab):
    r"""Apply the Moses Tokenizer/Detokenizer implemented in
     [sacremoses](https://github.com/alvations/sacremoses).

    .. note::
        sacremoses carries an LGPL 2.1+ license.

    Parameters
    ----------
    lang
        The language of the input.
    """

    def __init__(self, lang: str = 'en', vocab: Optional[Vocab] = None):
        self._lang = lang
        self._vocab = vocab
        if lang == 'zh':
            warnings.warn('You may not use MosesTokenizer for Chinese sentences because it is '
                          'not accurate. Try to use JiebaTokenizer. You may also tokenize the '
                          'chinese sentence to characters and learn a BPE.')
        self._tokenizer = sacremoses.MosesTokenizer(lang=lang)
        self._detokenizer = sacremoses.MosesDetokenizer(lang=lang)

        # Here, we need to warm-up the tokenizer to compile the regex
        # This will boost the performance in MacOS
        # For benchmarking results, see
        # https://gist.github.com/sxjscience/f59d2b88262fefd4fb08565c9dec6099
        self._warmup()

    def _warmup(self):
        _ = self.encode('hello world')
        _ = self.decode(['hello', 'world'])

    def encode(self, sentences, output_type=str):
        if output_type is str:
            if isinstance(sentences, list):
                return [self._tokenizer.tokenize(sentence, return_str=False)
                        for sentence in sentences]
            else:
                return self._tokenizer.tokenize(sentences, return_str=False)
        elif output_type is int:
            if self._vocab is None:
                raise TokenizerEncodeWithoutVocabError
            tokens = self.encode(sentences, str)
            if isinstance(sentences, list):
                return [self._vocab[ele_tokens] for ele_tokens in tokens]
            else:
                return self._vocab[tokens]
        else:
            raise NotImplementedError

    def encode_with_offsets(self, sentences, output_type=str):
        raise NotImplementedError('We cannot obtain the original offsets for MosesTokenizer.')

    def decode(self, tokens):
        is_multiple_sentences = is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = get_token_type(tokens)
        if token_type is str:
            ret = [self._detokenizer.detokenize(ele_tokens, return_str=True)
                   for ele_tokens in tokens]
        elif token_type is int:
            if self._vocab is None:
                raise TokenizerDecodeWithoutVocabError
            ret = [self._detokenizer.detokenize(self._vocab.to_tokens(ele_tokens), return_str=True)
                   for ele_tokens in tokens]
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
