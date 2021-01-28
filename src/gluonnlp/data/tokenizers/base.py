__all__ = ['BaseTokenizer', 'BaseTokenizerWithVocab', 'TOKENIZER_REGISTRY',
           'SentencesType', 'TokensType', 'TokenIDsType', 'TokenOffsetsType',
           'TokenizerEncodeWithoutVocabError', 'TokenizerDecodeWithoutVocabError',
           'TokenTypeNotSupportedError',
           'is_tokens_from_multiple_sentences', 'rebuild_offset_from_tokens',
           'get_token_type', 'get_char_offset_from_byte_offset',
           'create', 'create_with_json', 'list_all']

import abc
import itertools
from ..vocab import Vocab
from ...utils.registry import Registry
from typing import List, Tuple, Union, NewType, Optional


TOKENIZER_REGISTRY = Registry('Tokenizer')

SentencesType = NewType('SentencesType', Union[str, List[str]])
TokensType = NewType('TokensType', Union[List[str], List[List[str]]])
TokenIDsType = NewType('TokenIDsType', Union[List[int], List[List[int]]])
TokenOffsetsType = NewType('TokenOffsetsType', Union[List[Tuple[int, int]],
                                                     List[List[Tuple[int, int]]]])


class TokenizerEncodeWithoutVocabError(ValueError):
    """Error message when you want to encode sentence to a sequence of ids without
    specifying the vocabulary

    """
    def __init__(self):
        self.message = 'There is no vocab bound to the tokenizer. ' \
                       'Must set vocab if the output_type is "int". You may use' \
                       ' `tokenizer.set_vocab(vocab)` to attach the vocabulary.'
        super().__init__(self.message)


class TokenizerDecodeWithoutVocabError(ValueError):
    """Error message when you want to decode a sequence of ids to a string sentence without
    specifying the vocabulary

    """
    def __init__(self):
        self.message = 'Decode has "int" as the input token type. You must specify the ' \
                       'vocabulary in order to decode from integers. ' \
                       'You can use `tokenizer.set_vocab(vocab)`' \
                       ' to attach the vocabulary.'
        super().__init__(self.message)


class TokenTypeNotSupportedError(ValueError):
    def __init__(self, token_type):
        self.message =\
            'The token type is not supported, we only support ' \
            '"str" and "int" as the inner token types. Received type(token)="{}"'.format(token_type)
        super().__init__(self.message)


def is_tokens_from_multiple_sentences(tokens: Union[TokensType, TokenIDsType]) -> bool:
    """Return True if the tokens object consists of tokens from multiple sentences."""
    return len(tokens) > 0 and isinstance(tokens[0], list)


def get_token_type(tokens: Union[List[str], List[int], List[List[str]],
                                 List[List[int]]]) -> type:
    """

    Parameters
    ----------
    tokens
        The input tokens.

    Returns
    -------
    token_type
        If the tokens is empty, return `str`.
        Otherwise, return `str` if the input is str and `int` if the input is int.
    """
    if len(tokens) == 0:
        return str
    if isinstance(tokens[0], int):
        return int
    elif isinstance(tokens[0], str):
        return str
    elif isinstance(tokens[0], list):
        flatten_tokens_it = itertools.chain.from_iterable(tokens)
        try:
            first_token = next(flatten_tokens_it)
            return type(first_token)
        except StopIteration:
            return str
    else:
        raise TokenTypeNotSupportedError(type(tokens[0]))


def rebuild_offset_from_tokens(sentence: str, tokens: List[str]) \
        -> List[Tuple[int, int]]:
    """Recover the offset of the tokens in the original sentence.

    If you are using a subword tokenizer, make sure to remove the prefix/postfix of the tokens
    before using this function. Also, this does not work for n-gram-based (n>1) subword
    tokenization, i.e.
    it works for "gluonnlp" --> ["gluon", "nlp"]
    but not for "gluonnlp" --> ["gl", "lu", "uo", "on", "nl", "lp"]

    Parameters
    ----------
    sentence
        The input sentence
    tokens
        A list of strings that represent the tokenization result

    Returns
    -------
    offsets
        A list of start+end pairs: [(start0, end0), (start1, end1), ...].
        Each pair represents the start and end positions of the token in the original
        sentence.
    """
    running_offset = 0
    ret = []
    for token in tokens:
        token_offset = sentence.index(token, running_offset)
        token_len = len(token)
        running_offset = token_offset + token_len
        ret.append((token_offset, running_offset))
    return ret


def get_char_offset_from_byte_offset(sentence: str, byte_offsets: List[Tuple[int, int]]):
    """Get the character-level offsets based on the byte-level offsets

    Parameters
    ----------
    sentence
        The input sentence
    byte_offsets
        The byte-level offsets

    Returns
    -------
    char_offsets
        The character-level offsets
    """
    byte_offset_to_char_offset = {}
    byte_offset = 0
    for i, ele in enumerate(sentence):
        byte_offset_to_char_offset[byte_offset] = i
        byte_offset += len(ele.encode('utf-8'))
    byte_offset_to_char_offset[byte_offset] = i + 1  # Handle the last sentence
    ret = []
    for ele in byte_offsets:
        ret.append((byte_offset_to_char_offset[ele[0]],
                    byte_offset_to_char_offset[ele[1]]))
    return ret


class BaseTokenizer(abc.ABC):
    """Base class of the tokenizer"""
    @abc.abstractmethod
    def encode(self, sentences: SentencesType,
               output_type: type = str) \
            -> Union[TokensType, TokenIDsType]:
        """Encode the input sentence(s) into multiple tokens.

        Parameters
        ----------
        sentences
            The sentences to tokenize
        output_type
            The type of the output tokens.
            - str means each token is represented by its original text.
            - int means each token is represented by the index in the vocabulary.

        Returns
        -------
        tokens
            The output tokens.
        """
        pass

    @abc.abstractmethod
    def decode(self, tokens: Union[TokensType, TokenIDsType]) -> SentencesType:
        """Detokenize a sequence/multiple sequences of tokens to a single sentence/multiple
         sentences.

        Parameters
        ----------
        tokens
            The input tokens to decode

        Returns
        -------
        sentences
            The detokenized sentence(s)
        """
        pass

    def encode_with_offsets(self, sentences: SentencesType,
                            output_type: type = str) \
            -> Tuple[Union[TokensType, TokenIDsType], TokenOffsetsType]:
        """Encode the input sentence(s) into multiple tokens. Different from encode, it
        will also return the character start and end positions of each token in the original text.
        The original text is assumed to be

        Here, the default implementation is to use the tokenized result to recover the offsets.

        Parameters
        ----------
        sentences
            The sentence(s) to tokenize
        output_type
            The type of the output tokens.
            - `str` means each token is represented by its original text.
            - `int` means each token is represented by the index in the vocabulary.

        Returns
        -------
        tokens
            The output tokens.
        offsets
            The offsets of these tokens. Each encodes the start and end location in the original
            unicode string. We return the character-offset instead of the byte-offset.
        """
        raise NotImplementedError


class BaseTokenizerWithVocab(BaseTokenizer):
    """Base class of the tokenizer with vocabulary"""
    @property
    @abc.abstractmethod
    def vocab(self) -> Optional[Vocab]:
        """Get the vocab of the tokenizer

        Returns
        -------
        vocab
            The vocab of the tokenizer
        """
        pass

    @abc.abstractmethod
    def set_vocab(self, vocab: Vocab):
        """Set the vocab of the tokenizer"""
        pass


def create(name: str, *args, **kwargs) -> BaseTokenizer:
    """Create a tokenizer via name and additional arguments

    Parameters
    ----------
    name
    args
    kwargs

    Returns
    -------
    tokenizer
        The tokenizer

    Examples
    --------

    >>> import gluonnlp as nlp
    >>> tokenizer = nlp.tokenizers.create('whitespace')
    >>> moses_tokenizer = nlp.tokenizers.create('moses')
    """
    return TOKENIZER_REGISTRY.create(name, *args, **kwargs)


def create_with_json(name: str, json_str: str) -> BaseTokenizer:
    """Create tokenizer with name and a json string as the argument

    Parameters
    ----------
    name
    json_str

    Returns
    -------
    tokenizer
    """
    return TOKENIZER_REGISTRY.create_with_json(name, json_str)


def list_all() -> List[str]:
    """List the name of all the registered tokenizers

    Returns
    -------
    token_name_list
        Names of all the tokenizers
    """
    return TOKENIZER_REGISTRY.list_keys()
