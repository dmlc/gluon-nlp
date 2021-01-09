__all__ = ['YTTMTokenizer']

from typing import Optional, Tuple, Union, List
import os
from .base import *
from ..vocab import Vocab, load_vocab
from ...utils.lazy_imports import try_import_yttm


@TOKENIZER_REGISTRY.register('yttm')
class YTTMTokenizer(BaseTokenizerWithVocab):
    def __init__(self, model_path: str, vocab: Optional[Union[str, Vocab]] = None,
                 bpe_dropout: float = 0.0, n_threads: int = -1):
        """

        Parameters
        ----------
        model_path
            Path of the YTTM model file
        vocab_path
            Path of the vocabulary. It must be compatible to the YTTM model file.
            You are able to add new special tokens, but it must be compatible to the YTTM file.
        bpe_dropout
            The dropout probability in BPE-Dropout:
                "BPE-Dropout: Simple and Effective Subword Regularization"
        n_threads
            The number of threads for encoding
        """
        yttm = try_import_yttm()
        self._model_path = model_path
        self._bpe = yttm.BPE(model=model_path, n_threads=n_threads)
        self._bpe_dropout = bpe_dropout
        self._out_type = yttm.OutputType
        all_tokens = self._bpe.vocab()
        if vocab is not None:
            self._vocab = load_vocab(vocab)
        else:
            self._vocab = Vocab(all_tokens,
                                unk_token='<UNK>', pad_token='<PAD>',
                                bos_token='<BOS>', eos_token='<EOS>')
        self._meta_symbol = u'▁'  # U+2581 as the symbol for the first subword token
        # Assertions to verify that the vocabulary is compatible to the YTTM Model file.
        assert self._vocab.unk_token == '<UNK>' and \
               self._vocab.pad_token == '<PAD>' and \
               self._vocab.bos_token == '<BOS>' and \
               self._vocab.eos_token == '<EOS>', 'Incompatible vocabulary! Received ' \
                                                 'vocab={}'.format(self._vocab)
        ocab_nonspecial_tokens_set = frozenset(self._vocab.non_special_tokens)
        for token in all_tokens:
            if token not in [self._vocab.unk_token,
                             self._vocab.pad_token,
                             self._vocab.bos_token,
                             self._vocab.eos_token]:
                assert token in ocab_nonspecial_tokens_set
        for idx, token in enumerate(all_tokens):
            assert self._vocab[token] == idx
        self._first_subword_id_set = frozenset([self._vocab[ele]
                                                for ele in self._vocab.all_tokens
                                                if ele.startswith(self._meta_symbol)])

    def encode(self, sentences, output_type=str):
        is_single_sentence = not isinstance(sentences, list)
        if is_single_sentence:
            sentences = [sentences]
        if output_type is str:
            tokens = self._bpe.encode(sentences, output_type=self._out_type.SUBWORD,
                                      dropout_prob=self._bpe_dropout)
        elif output_type is int:
            tokens = self._bpe.encode(sentences, output_type=self._out_type.ID,
                                      dropout_prob=self._bpe_dropout)
        else:
            raise TokenTypeNotSupportedError(output_type)
        if is_single_sentence:
            return tokens[0]
        else:
            return tokens

    def decode(self, tokens):
        is_multi_sentences = is_tokens_from_multiple_sentences(tokens)
        token_type = get_token_type(tokens)
        if not is_multi_sentences:
            tokens = [tokens]
        if token_type is int:
            ret = self._bpe.decode(tokens)
        elif token_type is str:
            ret = []
            for ele_tokens in tokens:
                sentence = ''.join(ele_tokens)
                if len(sentence) > 0 and sentence[0] == self._meta_symbol:
                    sentence = sentence[1:]
                sentence = sentence.replace(self._meta_symbol, ' ')
                ret.append(sentence)
        else:
            raise TokenTypeNotSupportedError(token_type)
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
            encode_token = self._bpe.encode([sentence],
                                            output_type=self._out_type.SUBWORD,
                                            dropout_prob=self._bpe_dropout)[0]
            encode_id = self._bpe.encode([sentence],
                                         output_type=self._out_type.ID,
                                         dropout_prob=self._bpe_dropout)[0]
            encode_token_without_meta_symbol = [x.replace(self._meta_symbol, ' ')
                                                for x in encode_token]
            if len(encode_token_without_meta_symbol) > 0:
                encode_token_without_meta_symbol[0] = \
                    encode_token_without_meta_symbol[0].replace(' ', '')
            encode_offset = rebuild_offset_from_tokens(sentence, encode_token_without_meta_symbol)
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

    def is_first_subword(self, tokens: Union[str, int, List[str], List[int]]) \
            -> Union[bool, List[bool]]:
        """Whether the token is the first subword token in a list of subword tokens

        Parameters
        ----------
        tokens
            The input tokens

        Returns
        -------
        ret
            Whether the token is the first subword token in a sequence of subword tokens
            that construct the token
        """
        if isinstance(tokens, str):
            return tokens.startswith(self._meta_symbol)
        elif isinstance(tokens, int):
            return tokens in self._first_subword_id_set
        elif isinstance(tokens, list):
            if len(tokens) == 0:
                return []
            if isinstance(tokens[0], str):
                return [ele.startswith(self._meta_symbol) for ele in tokens]
            elif isinstance(tokens[0], int):
                return [ele in self._first_subword_id_set for ele in tokens]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        raise NotImplementedError('Cannot set vocabulary for the YTTMTokenizer.')

    def set_bpe_dropout(self, bpe_dropout: float):
        """Set the bpe dropout probability

        Parameters
        ----------
        bpe_dropout
            The dropout ratio for BPE Dropout
        """
        self._bpe_dropout = bpe_dropout

    def __repr__(self):
        ret = '{}(\n' \
              '   model_path = {}\n' \
              '   bpe_dropout = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._model_path),
                         self._bpe_dropout,
                         self._vocab)
        return ret

    def __getstate__(self):
        """Support multiprocessing by making it pickleble"""
        state = self.__dict__.copy()
        state['_bpe'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        yttm = try_import_yttm()
        self._bpe = yttm.BPE(self._model_path)
