__all__ = ['WhitespaceTokenizer']

from typing import Optional
from .base import TOKENIZER_REGISTRY, BaseTokenizerWithVocab, TokenizerEncodeWithoutVocabError,\
    TokenizerDecodeWithoutVocabError, TokenTypeNotSupportedError,\
    rebuild_offset_from_tokens, is_tokens_from_multiple_sentences, get_token_type
from ..vocab import Vocab


@TOKENIZER_REGISTRY.register('whitespace')
class WhitespaceTokenizer(BaseTokenizerWithVocab):
    """Basic tokenizer that tokenizes the input via white spaces."""
    def __init__(self, vocab: Optional[Vocab] = None):
        self._vocab = vocab

    def encode(self, sentences, output_type=str):
        is_multiple_sentences = isinstance(sentences, list)
        if not is_multiple_sentences:
            sentences = [sentences]
        if output_type is str:
            tokens = [sentence.split() for sentence in sentences]
        elif output_type is int:
            if self._vocab is None:
                raise TokenizerEncodeWithoutVocabError
            tokens = [self._vocab[sentence.split()] for sentence in sentences]
        else:
            raise NotImplementedError
        if is_multiple_sentences:
            return tokens
        else:
            return tokens[0]

    def encode_with_offsets(self, sentences, output_type=str):
        if output_type is int and self.vocab is None:
            raise TokenizerEncodeWithoutVocabError
        if output_type not in [int, str]:
            raise TokenTypeNotSupportedError(output_type)
        is_multiple_sentences = isinstance(sentences, list)
        if not is_multiple_sentences:
            sentences = [sentences]
        all_tokens = self.encode(sentences, output_type=str)
        offsets = []
        for ele_tokens, ele_sentence in zip(all_tokens, sentences):
            ele_offsets = rebuild_offset_from_tokens(ele_sentence, ele_tokens)
            offsets.append(ele_offsets)
        if is_multiple_sentences:
            return all_tokens, offsets
        else:
            return all_tokens[0], offsets[0]

    def decode(self, tokens):
        is_multiple_sentences = is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = get_token_type(tokens)
        if token_type is str:
            ret = [' '.join(ele_tokens) for ele_tokens in tokens]
        elif token_type is int:
            if self._vocab is None:
                raise TokenizerDecodeWithoutVocabError
            ret = [' '.join(self.vocab.to_tokens(ele_tokens)) for ele_tokens in tokens]
        else:
            raise TokenTypeNotSupportedError(token_type)
        if is_multiple_sentences:
            return ret
        else:
            return ret[0]

    @property
    def vocab(self) -> Vocab:
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        """Set the vocabulary of the tokenizer

        Parameters
        ----------
        vocab
        """
        self._vocab = vocab
