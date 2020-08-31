__all__ = ['SubwordNMTTokenizer']

import os
from typing import Optional, Union, List
from .base import TOKENIZER_REGISTRY, BaseTokenizerWithVocab,\
    TokenizerEncodeWithoutVocabError, TokenizerDecodeWithoutVocabError,\
    TokenTypeNotSupportedError, rebuild_offset_from_tokens,\
    TokensType, TokenIDsType, TokenOffsetsType, SentencesType, is_tokens_from_multiple_sentences, \
    get_token_type
from ..vocab import Vocab, load_vocab
from ...utils.lazy_imports import try_import_subword_nmt


@TOKENIZER_REGISTRY.register('subword_nmt')
class SubwordNMTTokenizer(BaseTokenizerWithVocab):
    def __init__(self, model_path, vocab: Union[str, Vocab],
                 separator: str = '@@', bpe_dropout: float = 0.0,
                 suffix: str = '</w>'):
        """

        Parameters
        ----------
        model_path
        vocab
        separator
        bpe_dropout
        suffix
        """
        try_import_subword_nmt()
        from subword_nmt.apply_bpe import BPE
        self._model_path = model_path
        self._vocab = load_vocab(vocab)
        self._separator = separator
        self._bpe_dropout = bpe_dropout
        self._suffix = suffix
        with open(self._model_path, 'r', encoding='utf-8') as merge_codes:
            self._bpe = BPE(codes=merge_codes, separator=self._separator)
        self._last_subword_id_set = frozenset([self._vocab[ele]
                                               for ele in self._vocab.all_tokens
                                               if not ele.endswith(self._separator)])

    def transform_sentence(self, sentence):
        """replace the separator in encoded result with suffix

        a@@, b@@, c ->  a, b, c</w>

        Parameters
        ----------
        sentence

        Returns
        -------
        new_sentence
        """
        return [word[:-2] if len(word) > 2 and word[-2:] == self._separator else word + self._suffix
                for word in sentence]

    def encode(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        if output_type is str:
            ret = [self.transform_sentence(
                self._bpe.segment(sentence, dropout=self._bpe_dropout).split(' '))
                   for sentence in sentences]
        elif output_type is int:
            if self._vocab is None:
                raise TokenizerEncodeWithoutVocabError
            ret = [self._vocab[self.transform_sentence(
                self._bpe.segment(sentence, dropout=self._bpe_dropout).split(' '))]
                   for sentence in sentences]
        else:
            raise TokenTypeNotSupportedError(output_type)
        if is_multi_sentences:
            return ret
        else:
            return ret[0]

    def encode_with_offsets(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        tokens = []
        token_ids = []
        offsets = []
        for sentence in sentences:
            encode_token = self.transform_sentence(
                self._bpe.segment(sentence, dropout=self._bpe_dropout).split(' '))
            encode_id = self._vocab[encode_token]
            encode_token_without_suffix = [x.replace(self._suffix, '') for x in encode_token]
            encode_offset = rebuild_offset_from_tokens(sentence, encode_token_without_suffix)
            tokens.append(encode_token)
            token_ids.append(encode_id)
            offsets.append(encode_offset)
        if not is_multi_sentences:
            tokens = tokens[0]
            token_ids = token_ids[0]
            offsets = offsets[0]
        if output_type is str:
            return tokens, offsets
        elif output_type is int:
            return token_ids, offsets
        else:
            raise TokenTypeNotSupportedError(output_type)

    def decode(self, tokens: Union[TokensType, TokenIDsType]) -> SentencesType:
        is_multiple_sentences = is_tokens_from_multiple_sentences(tokens)
        if not is_multiple_sentences:
            tokens = [tokens]
        token_type = get_token_type(tokens)
        if token_type is str:
            ret = [''.join(ele_tokens).replace(self._suffix, ' ').strip()
                   for ele_tokens in tokens]
        elif token_type is int:
            if self._vocab is None:
                raise TokenizerDecodeWithoutVocabError
            ret = [''.join(self._vocab.to_tokens(ele_tokens)).replace(self._suffix, ' ').strip()
                   for ele_tokens in tokens]
        else:
            raise TokenTypeNotSupportedError(token_type)
        if is_multiple_sentences:
            return ret
        else:
            return ret[0]

    def is_last_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        """Whether the token is the last subword token. This can be used
        for whole-word masking.

        Parameters
        ----------
        tokens
            The input tokens

        Returns
        -------
        ret
            Whether the token is the last subword token in the list of subwords
        """
        if isinstance(tokens, str):
            return not tokens.endswith(self._separator)
        elif isinstance(tokens, int):
            return tokens in self._last_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [not ele.endswith(self._separator) for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._last_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def vocab(self) -> Optional[Vocab]:
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        self._vocab = vocab

    def set_bpe_dropout(self, bpe_dropout: float):
        self._bpe_dropout = bpe_dropout

    def __repr__(self):
        ret = '{}(\n' \
              '   model_path = {}\n' \
              '   separator = {}\n' \
              '   bpe_dropout = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._model_path),
                         self._separator,
                         self._bpe_dropout,
                         self._vocab)
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_bpe'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        from subword_nmt.apply_bpe import BPE
        with open(self._model_path, 'r', encoding='utf-8') as merge_codes:
            self._bpe = BPE(codes=merge_codes, separator=self._separator)
