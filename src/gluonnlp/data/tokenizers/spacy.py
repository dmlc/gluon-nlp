__all__ = ['SpacyTokenizer']

from typing import Optional
from .base import TOKENIZER_REGISTRY, BaseTokenizerWithVocab,\
    TokenizerEncodeWithoutVocabError, TokenTypeNotSupportedError, SentencesType
from ..vocab import Vocab
from ...utils.lazy_imports import try_import_spacy


@TOKENIZER_REGISTRY.register('spacy')
class SpacyTokenizer(BaseTokenizerWithVocab):
    r"""Wrapper of the Spacy Tokenizer.

    Users of this class are required to install `spaCy <https://spacy.io/usage/>`_
    and download corresponding NLP models, such as :samp:`python -m spacy download en`.

    Only spacy>=2.0.0 is supported.

    Parameters
    ----------
    lang
        The language of the input. If we just specify the lang and do not specify the model,
        we will provide the tokenizer with pre-selected models.
    model
        The language to tokenize. Default is None, and we will choose the tokenizer
        automatically based on the language:

        - en --> 'en_core_web_sm'
        - de --> 'de_core_news_sm'
        - fr --> 'fr_core_news_sm'
        - ja --> 'ja_core_news_sm'

        For more details about how to set this flag, you may refer to
        https://spacy.io/usage/models for supported languages.

        Also, you may refer to https://github.com/explosion/spacy-models/blob/master/compatibility.json

    vocab
        The vocabulary of the tokenizer. Can be optional. You must specify this if you will need to map
        the raw text into integers.

    Examples
    --------
    >>> import gluonnlp
    >>> tokenizer = gluonnlp.data.SpacyTokenizer()
    >>> tokenizer.encode('Gluon NLP toolkit provides a suite of text processing tools.')
    ['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of', 'text', 'processing', 'tools', '.']
    >>> tokenizer = gluonnlp.data.SpacyTokenizer('de')
    >>> tokenizer.encode('Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools'
    ...                  ' zur Verfügung.')
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
    'zur', 'Verfügung', '.']
    >>> tokenizer = gluonnlp.data.SpacyTokenizer(model='de_core_news_sm')
    >>> tokenizer.encode('Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools'
    ...                  ' zur Verfügung.')
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
    'zur', 'Verfügung', '.']

    """

    def __init__(self, lang: Optional[str] = 'en', model: Optional[str] = None,
                 vocab: Optional[Vocab] = None):
        self._vocab = vocab
        self._model = model
        spacy = try_import_spacy()
        if model is None:
            assert lang is not None
            if lang == 'en':
                model = 'en_core_web_sm'
            elif lang == 'de':
                model = 'de_core_news_sm'
            elif lang == 'fr':
                model = 'fr_core_news_sm'
            elif lang == 'ja':
                model = 'ja_core_news_cm'
            else:
                model = 'xx_ent_wiki_sm'
        self._model = model
        retries = 5
        try:
            self._nlp = spacy.load(model, disable=['parser', 'tagger', 'ner'])
        except (Exception, IOError, OSError):
            from spacy.cli import download
            while retries >= 0:
                try:
                    download(model, False)
                    self._nlp = spacy.load(model, disable=['parser', 'tagger', 'ner'])
                    break
                except (Exception, IOError, OSError) as download_err:
                    retries -= 1
                    if retries < 0:
                        print('SpaCy Model for the specified model="{model}" has not been '
                              'successfully loaded. You need to check the installation guide in '
                              'https://spacy.io/usage/models. Usually, the installation command '
                              'should be `python3 -m spacy download {model}`.\n'
                              'Complete Error Message: {err_msg}'.format(model=model,
                                                                         err_msg=str(download_err)))
                        raise

    def encode(self, sentences, output_type=str):
        if output_type is str:
            if isinstance(sentences, list):
                return [[tok.text for tok in self._nlp(sentence)] for sentence in sentences]
            else:
                return [tok.text for tok in self._nlp(sentences)]
        elif output_type is int:
            if self._vocab is None:
                raise TokenizerEncodeWithoutVocabError
            tokens = self.encode(sentences, str)
            if isinstance(sentences, list):
                return [self._vocab[ele_tokens] for ele_tokens in tokens]
            else:
                return [self._vocab[tokens]]
        else:
            raise TokenTypeNotSupportedError(output_type)

    def encode_with_offsets(self, sentences: SentencesType, output_type=str):
        is_multi_sentences = isinstance(sentences, list)
        if not is_multi_sentences:
            sentences = [sentences]
        all_tokens = [self._nlp(sentence) for sentence in sentences]
        offsets = [[(tok.idx, tok.idx + len(tok.text)) for tok in tokens]
                   for tokens in all_tokens]
        if output_type is str:
            out_tokens = [[tok.text for tok in tokens] for tokens in all_tokens]
        elif output_type is int:
            if self._vocab is None:
                raise TokenizerEncodeWithoutVocabError
            out_tokens = [self._vocab[[tok.text for tok in tokens]] for tokens in all_tokens]
        else:
            raise TokenTypeNotSupportedError
        if is_multi_sentences:
            return out_tokens, offsets
        else:
            return out_tokens[0], offsets[0]

    def decode(self, tokens):
        raise NotImplementedError(
            'We decide not to implement the decode feature for SpacyTokenizer'
            ' because detokenization is not well-supported by'
            ' spacy. For more details, you may refer to the stack-overflow discussion:'
            ' https://stackoverflow.com/questions/50330455/how-to-detokenize-spacy-text-without-doc-context. '
            'Also, we welcome your contribution for adding a reasonable detokenizer for SpaCy.')

    @property
    def model(self):
        return self._model

    @property
    def vocab(self):
        return self._vocab

    def set_vocab(self, vocab: Vocab):
        """Set the vocabulary of the tokenizer

        Parameters
        ----------
        vocab
            Update the inner vocabulary of the tokenizer to the given vocabulary.
        """
        self._vocab = vocab
