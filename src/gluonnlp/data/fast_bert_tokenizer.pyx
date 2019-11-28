"""Used to tokenize text for use with a BERT model."""

from typing import List, Dict, Tuple
import unicodedata


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    return text.strip().split()


cdef class BasicTokenizer:
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""
    cdef public bint lower
    
    def __init__(self, lower=True):
        """Constructs a BasicTokenizer.
    
        :param do_lower_case: Whether to lower case the input.
        """
        self.lower = lower
    
    def tokenize(self, text) -> List[str]:
        """Tokenizes a piece of text."""
        # Developments notes:
        #   - The original BERT code loops over every char in pure Python 4 times
        #     (several more times if you include loops that are happening inside built-ins).
        #     This optimized version uses generators and only loops over each char explicitly twice.
        #   - This runs in two separate steps because I thought it would be better to apply
        #     `lower` and do accent normalization on the whole string at once rather than parts.
        #     In Python this limits the amount of looping so it provides a speedup.  But in Cython
        #     that may not actually be true.
        
        # Step 1: normalize whitespace, filter control characters, and add spaces around
        #   Chinese characters.
        step1_text = "".join(_step1(text)).strip()
        if self.lower:
            step1_text = step1_text.lower()
        
        # Normalize unicode characters to strip accents
        # This isn't part of either step1 or step2 because it runs on the entire
        # string and any looping over chars takes place in a built-in C loop
        # that is likely more optimized than anything that I can write here.
        step1_text = unicodedata.normalize("NFD", step1_text)
        
        # Step 2: filter non-spacing marks (Mn unicode category) and
        #   add spaces around any punctuation.
        # This is pretty simple in comparison to the other step.
        output_tokens = "".join(_step2(step1_text)).split()
        return output_tokens


cdef class WordpieceTokenizer:
    """Runs WordPiece tokenziation."""
    
    cdef public vocab
    cdef public str unk_token
    cdef public long max_input_chars_per_word
    
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
    
    def tokenize(self, text) -> List[str]:
        """Tokenizes a piece of text into its word pieces.
    
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
    
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
    
        :param text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        :returns: A list of wordpiece tokens.
        """
        cdef long max_input_chars_per_word = self.max_input_chars_per_word
        cdef:
            bint is_bad
            long start
            long end
            Py_ssize_t n_chars
        
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            n_chars = len(chars)
            if n_chars > max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            
            is_bad = False
            start = 0
            sub_tokens = []
            while start < n_chars:
                end =  n_chars
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        # Now it's a subword
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _step1(str text):
    """First step in pre-processing test for BERT.
    
    This function yields unicode characters while, normalizing all whitespace to spaces,
    filtering control characters, and adding spaces around chinese characters.
    """
    cdef bint prev_ch_whitespace = False
    cdef str ch
    cdef str cat
    cdef Py_UCS4 cp
    
    for ch in text:
        cp = <Py_UCS4>ch  # Casting this here removes the need for some extra error checking in the loop.
        
        # `is_control` used unicodedata.category for every character that's not \t, \n, or \r
        # which is basically everything. So it's better to just call it on everything
        # to begin with and pass the result around.
        cat = unicodedata.category(ch)
        if cp == 0 or cp == 0xfffd or _is_control(cp, cat):
            continue
        if _is_whitespace(cp, cat):
            yield " "
            prev_ch_whitespace = True
        else:
            # From the original BERT code:
            # ---------------------------
            # This was added on November 1st, 2018 for the multilingual and Chinese
            # models. This is also applied to the English models now, but it doesn't
            # matter since the English models were not trained on any Chinese data
            # and generally don't have any Chinese data in them (there are Chinese
            # characters in the vocabulary because Wikipedia does have some Chinese
            # words in the English Wikipedia.).
           
            # NB: Our regression tests will fail if we get rid of this because
            #   our dev datasets have chinese characters in them.
            #   I have no idea if this is important for production or not
            if _is_chinese_char(cp):
                # Add whitespace around any CJK character.
                if not prev_ch_whitespace:
                    yield " "
                yield ch
                yield " "
            else:
                yield ch
                prev_ch_whitespace = False


def _step2(str text):
    """After encoding normalization, whitespace normalization, chinese character normalization,
    and accent stripping, this step runs and filters non-spacing marks (Mn unicode category) and
    adds spaces around any punctuation.
    """
    cdef str ch
    cdef str cat

    for ch in text:
        cat = unicodedata.category(ch)
        # Filter some chars (non-spacing mark)
        if cat == "Mn":
            continue
        # Add whitespace around any punctuation
        if _is_punctuation(ch, cat):
            yield " "
            yield ch
            yield " "
        else:
            yield ch


cdef inline bint _is_punctuation(Py_UCS4 cp, str cat):
    """Checks whether `cp` is a punctuation character.
    
    We treat all non-letter/number ASCII as punctuation.
    Characters such as "^", "$", and "`" are not in the Unicode
    Punctuation class but we treat them as punctuation anyways, for
    consistency.
    """
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    if cat.startswith("P"):
        return True
    return False


cdef inline bint _is_control(Py_UCS4 ch, str cat):
    """Checks whether `ch` is a control character."""
    # Some of these are technically control characters but we count them as whitespace
    if ch == u"\t" or ch == u"\n" or ch == u"\r":
        return False
    if cat in ("Cc", "Cf"):
        return True
    return False


cdef inline bint _is_whitespace(Py_UCS4 ch, str cat):
    """Checks whether `chars` is a whitespace character.
    
    \t, \n, and \r are technically control characters but we treat them
    as whitespace since they are generally considered as such.
    """
    if ch == u" " or ch == u"\t" or ch == u"\n" or ch == u"\r":
        return True
    if cat == "Zs":
        return True
    return False


cdef inline bint _is_chinese_char(Py_UCS4 cp):
    """Checks whether CP is the codepoint of a CJK character.
    
    This defines a "chinese character" as anything in the CJK Unicode block:
      https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)

    Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    despite its name. The modern Korean Hangul alphabet is a different block,
    as is Japanese Hiragana and Katakana. Those alphabets are used to write
    space-separated words, so they are not treated specially and handled
    like the all of the other languages.
    """
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


# Public functions for testing
def is_punctuation(Py_UCS4 cp, str cat):
    return _is_punctuation(cp, cat)

def is_control(Py_UCS4 ch, str cat):
    return _is_control(ch, cat)

def is_whitespace(Py_UCS4 ch, str cat):
    return _is_whitespace(ch, cat)

def is_chinese_char(Py_UCS4 cp):
    return _is_chinese_char(cp)

