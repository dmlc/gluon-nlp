import re
import regex
import requests
import unicodedata
from typing import List, Pattern, Union, Tuple, Optional
from sacremoses.normalize import MosesPunctNormalizer


non_printing_char_regex = regex.compile(r'\p{C}')


class MosesNormalizer:
    """Normalizes the input sentence. Currently, we support the combination of the

    Moses Punctuation Normalizer 'normalize-punctuation.perl' and the
     'remove-non-printing-char.perl' in [mosesdecoder](https://github.com/moses-smt/mosesdecoder):

    Also, we will normalize the

    Parameters
    ----------
    lang
        The input language
    remove_non_printable_char
        Whether to remove the non-printable unicode characters in the input
    unicode_norm_form
        The unicode normalization format used. Supported

    """
    def __init__(self, lang: str, remove_non_printable_char: bool = True,
                 unicode_norm_form: Optional[str] = None):
        self._remove_non_printable_char = remove_non_printable_char
        self._moses_normalizer = MosesPunctNormalizer(lang)
        self._unicode_norm_form = unicode_norm_form
        if unicode_norm_form is not None:
            assert unicode_norm_form in ['NFC', 'NFKC', 'NFD', 'NFKD'],\
                'Unsupported unicode normalization format, you may refer to ' \
                'https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize for ' \
                'more details.'
        self.__warmup()

    def __warmup(self):
        self('hello world')

    def __call__(self, sentence: str) -> str:
        if self._unicode_norm_form:
            sentence = unicodedata.normalize(self._unicode_norm_form, sentence)
        sentence = self._moses_normalizer.normalize(sentence)
        if self._remove_non_printable_char:
            return non_printing_char_regex.sub(' ', sentence)
        else:
            return sentence


def _words_match_regex(words: List[str], ignore_case=False, replace_white_space=False) -> Pattern:
    """Obtain the regex that finds whether a given corpus contains any word in the input words

    Parameters
    ----------
    words

    Returns
    -------
    regex

    """
    words = [ele for ele in words if ele]
    if ignore_case:
        flags = re.IGNORECASE
    else:
        flags = 0
    if replace_white_space:
        words = [ele.replace(' ', r'\s+') for ele in words]
    regex = re.compile('[^a-z]({words})[^a-z]|^({words})$|^({words})[^a-z]|[^a-z]({words})$'
                       .format(words='|'.join(words)), flags)
    return regex


class ProfanityFilter:
    """Detect whether the corpus contains possible profanity content.

    We use the word list from https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

    """
    def __init__(self, langs: Optional[Union[str, List, Tuple]] = None):
        def _download(url, retries=5):
            while retries + 1 > 0:
                try:
                    r = requests.get(url, stream=True, verify=True)
                    if r.status_code != 200:
                        raise RuntimeError('Failed downloading url {}'.format(url))
                    return r.text
                except Exception as e:
                    retries -= 1
                    if retries <= 0:
                        raise e
                    print('download failed due to {}, retrying, {} attempt{} left'
                          .format(repr(e), retries, 's' if retries > 1 else ''))
        url_path =\
            'https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/b36ce5c34c14cb7872dd4c2a4e55fe526138462d/{lang}'
        available_langs = {'ar', 'cs', 'da', 'de', 'en', 'eo', 'es', 'fa', 'fi', 'fr', 'hi', 'hu',
                           'it', 'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tlh', 'tr',
                           'zh'}
        self._suspicious_words = []
        if langs is None:
            filter_langs = available_langs
        elif isinstance(langs, str):
            filter_langs = [langs]
        elif isinstance(langs, (tuple, list)):
            filter_langs = list(langs)
        else:
            raise ValueError('Unsupported input langs={}'.format(langs))
        for lang in filter_langs:
            assert lang in available_langs, \
                'lang={} is not supported. All supported languages={}'.format(lang, available_langs)
            out = _download(url_path.format(lang=lang))
            self._suspicious_words += [word.strip() for word in out.split('\n') if word.strip()]
        self._regex = _words_match_regex(self._suspicious_words)

    def match(self, corpus: str) -> bool:
        """Search whether the input corpus contains the suspicious bad words.

        Parameters
        ----------
        corpus
            Input string

        Returns
        -------
        ret
            Whether the input corpus contains profanity words.
        """
        return self._regex.match(corpus) is not None
