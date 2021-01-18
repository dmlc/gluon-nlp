__all__ = ['SentencepieceTokenizer']

import os
from typing import Optional, Union, List
from .base import *
from ..vocab import Vocab, load_vocab
from ...utils.lazy_imports import try_import_sentencepiece


@TOKENIZER_REGISTRY.register('spm')
class SentencepieceTokenizer(BaseTokenizerWithVocab):
    r"""Apply the Sentencepiece Tokenizer, which trains subword tokenization via the
    unigram language modeling.

    Users of this class are required to `install sentencepiece
    <https://github.com/google/sentencepiece>`_. For example, one can use
    :samp:`pip install sentencepiece`


    Parameters
    ----------
    model_path
        Path to the pre-trained sentencepiece model.
    vocab
        Path to the vocabulary of the sentencepiece model in GluonNLP
    num_best
        A scalar for sampling subwords. If num_best = {0,1}, no sampling is performed.
        If num_best > 1, then samples from the num_best results.
        If num_best < 0, then assume that num_best is infinite and
        samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling
        algorithm.
    alpha
        A scalar for a smoothing parameter for probability rescaling.
    lowercase
        Whether to convert the input string to lower-case strings
    **kwargs

    Examples
    --------
    >>> from mxnet import gluon
    >>> url = 'https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com/tokenizer_test_models/sentencepiece/test_ende-a9bee4.model'
    >>> model_f = gluon.utils.download(url)
    -etc-

    >>> tokenizer = gluonnlp.data.SentencepieceTokenizer(model_f)
    >>> sentence = 'This is a very awesome, life-changing sentence.'
    >>> tokenizer.encode(sentence)
    ['▁This', '▁is', '▁a', '▁very', '▁awesome', ',', '▁life', '-', 'ch', 'anging', '▁sentence', '.']

    >>> tokenizer.decode(tokenizer.encode(sentence))
    'This is a very awesome, life-changing sentence.'

    >>> os.remove('test_ende-a9bee4.model')

    """

    def __init__(self, model_path: Optional[str] = None,
                 vocab: Optional[Union[str, Vocab]] = None,
                 nbest: int = 0, alpha: float = 0.0, lowercase=False,
                 **kwargs):
        self._model_path = model_path
        sentencepiece = try_import_sentencepiece()
        from ...third_party.sentencepiece_pb2 import SentencePieceText
        self._spt_cls = SentencePieceText
        self._sp_model = sentencepiece.SentencePieceProcessor()
        self._sp_model.load(model_path)
        self._nbest = nbest
        self._alpha = alpha
        self._lowercase = lowercase
        self._meta_symbol = u'▁'
        sp_model_all_tokens = [self._sp_model.id_to_piece(i) for i in range(len(self._sp_model))]
        special_tokens_kv = dict()
        existing_control_token_ids = set()
        token_id_to_token_name = dict()
        if self._sp_model.unk_id() != -1:
            special_tokens_kv['unk_token'] = self._sp_model.id_to_piece(self._sp_model.unk_id())
            existing_control_token_ids.add(self._sp_model.unk_id())
            token_id_to_token_name[self._sp_model.unk_id()] = 'unk_token'
        if self._sp_model.pad_id() != -1:
            special_tokens_kv['pad_token'] = self._sp_model.id_to_piece(self._sp_model.pad_id())
            existing_control_token_ids.add(self._sp_model.pad_id())
            token_id_to_token_name[self._sp_model.pad_id()] = 'pad_token'
        if self._sp_model.bos_id() != -1:
            special_tokens_kv['bos_token'] = self._sp_model.id_to_piece(self._sp_model.bos_id())
            existing_control_token_ids.add(self._sp_model.bos_id())
            token_id_to_token_name[self._sp_model.bos_id()] = 'bos_token'
        if self._sp_model.eos_id() != -1:
            special_tokens_kv['eos_token'] = self._sp_model.id_to_piece(self._sp_model.eos_id())
            existing_control_token_ids.add(self._sp_model.eos_id())
            token_id_to_token_name[self._sp_model.eos_id()] = 'eos_token'
        existing_control_tokens = set([self._sp_model.id_to_piece(ele)
                                       for ele in existing_control_token_ids])
        other_control_tokens_ids = \
            [i for i in range(len(self._sp_model))
             if self._sp_model.is_control(i) and i not in existing_control_token_ids]
        other_control_tokens = set([self._sp_model.id_to_piece(ele)
                                    for ele in other_control_tokens_ids])
        matched_other_control_tokens = dict()
        for k, v in kwargs.items():
            if k in special_tokens_kv:
                if v != special_tokens_kv[k]:
                    raise ValueError(
                        '"vocab.{}" is already set to "{}" in the sentencepiece model. '
                        'Cannot reset it to "{}"'.format(k, special_tokens_kv[k], v))
                continue
            if v in existing_control_tokens:
                if k != token_id_to_token_name[v]:
                    raise ValueError('"{}" is already registered as "vocab.{}". '
                                     'We cannot rename it to "vocab.{}".'
                                     .format(v, token_id_to_token_name[v], k))
                continue
            if v in other_control_tokens:
                if v in matched_other_control_tokens:
                    raise ValueError(
                        '"{}" has already been registered as "vocab.{}", '
                        'we cannot register it again as "vocab.{}".'
                            .format(v, matched_other_control_tokens[v], k))
                matched_other_control_tokens[v] = k
                special_tokens_kv[k] = v
            else:
                raise ValueError('Mismatch vocabulary! All special tokens specified '
                                 'must be control tokens in the sentencepiece vocabulary.')
        if vocab is None:
            if len(matched_other_control_tokens) < len(other_control_tokens):
                for i, token in enumerate(other_control_tokens.difference(
                        set(matched_other_control_tokens.keys()))):
                    token_key = 'other{}_token'.format(i)
                    assert token_key not in special_tokens_kv
                    special_tokens_kv[token_key] = token
            self._vocab = Vocab(sp_model_all_tokens, **special_tokens_kv)
        else:
            self._vocab = load_vocab(vocab)
        # Sanity check
        assert len(self._vocab.all_tokens) >= len(sp_model_all_tokens)
        assert self._vocab.all_tokens[:len(sp_model_all_tokens)] == sp_model_all_tokens
        for token in self._vocab.special_tokens:
            piece_id = self._sp_model.piece_to_id(token)
            if not self._sp_model.is_unknown(piece_id):
                assert self._sp_model.is_control(piece_id), \
                    'Vocab mismatch! "{}" is a special token in the given vocab but not in the ' \
                    'sentencepiece model!'.format(token)
        self._first_subword_id_set = frozenset([self._vocab[ele]
                                                for ele in sp_model_all_tokens
                                                if ele.startswith(self._meta_symbol)])

    def encode(self, sentences, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        if self._lowercase:
            sentences = [sentence.lower() for sentence in sentences]
        if output_type is str:
            ret = [self._sp_model.sample_encode_as_pieces(sentence, self._nbest, self._alpha)
                   for sentence in sentences]
        elif output_type is int:
            ret = [self._sp_model.sample_encode_as_ids(sentence, self._nbest, self._alpha)
                   for sentence in sentences]
        else:
            raise TokenTypeNotSupportedError(output_type)
        if is_multi_sentences:
            return ret
        else:
            return ret[0]

    def decode(self, tokens):
        is_multi_sentences = is_tokens_from_multiple_sentences(tokens)
        token_type = get_token_type(tokens)
        if not is_multi_sentences:
            tokens = [tokens]
        if token_type is str:
            ret = [self._sp_model.decode_pieces(ele_tokens) for ele_tokens in tokens]
        elif token_type is int:
            ret = [self._sp_model.decode_ids(ele_tokens) for ele_tokens in tokens]
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
            if self._lowercase:
                sentence = sentence.lower()
            spt = self._spt_cls()
            spt.ParseFromString(self._sp_model.SampleEncodeAsSerializedProto(
                sentence, self._nbest, self._alpha))
            tokens.append([ele.piece for ele in spt.pieces])
            token_ids.append([ele.id for ele in spt.pieces])
            # In theory, we can recover the character offset from byte offset
            sentence_byte_offsets = [(ele.begin, ele.end) for ele in spt.pieces]
            char_offsets = get_char_offset_from_byte_offset(sentence, sentence_byte_offsets)
            offsets.append(char_offsets)
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
        """Whether the token is the first subword token. This can be used to implement
        whole-word masking.

        Parameters
        ----------
        tokens
            The input tokens

        Returns
        -------
        ret
            Whether the token is the first subword token in the list of subwords
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

    def set_vocab(self, vocab):
        raise NotImplementedError('Currently, we cannot set the vocabulary of a '
                                  'SentencepieceTokenizer.')

    @property
    def lowercase(self):
        return self._lowercase

    def set_subword_regularization(self, nbest, alpha):
        """Set the subword-regularization parameters

        For more details, you may refer to the official SentencePiece library:

        https://github.com/google/sentencepiece

        Parameters
        ----------
        nbest
        alpha

        Returns
        -------

        """
        self._nbest = nbest
        self._alpha = alpha

    def __repr__(self):
        ret = '{}(\n' \
              '   model_path = {}\n' \
              '   lowercase = {}, nbest = {}, alpha = {}\n' \
              '   vocab = {}\n' \
              ')'.format(self.__class__.__name__,
                         os.path.realpath(self._model_path),
                         self._lowercase, self._nbest, self._alpha,
                         self._vocab)
        return ret

    def __getstate__(self):
        """Make the SentencepieceTokenizer pickleble.
         We will remove the _spt_cls and _sp_model, which are not picklable, and try to
         reconstruct the class via the saved model_path. This behavior is only acceptable for
         multiprocessing and should not be used to save sentencepiece models."""
        state = self.__dict__.copy()
        state['_spt_cls'] = None
        state['_sp_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        sentencepiece = try_import_sentencepiece()
        from ...third_party.sentencepiece_pb2 import SentencePieceText
        self._spt_cls = SentencePieceText
        self._sp_model = sentencepiece.SentencePieceProcessor()
        ret = self._sp_model.load(self._model_path)
        assert ret is True, 'Cannot load data from the saved seralized protobuffer!'
