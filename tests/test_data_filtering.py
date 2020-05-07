from gluonnlp.data.filtering import ProfanityFilter, MosesNormalizer


def test_profanity_filter():
    profanity_filter = ProfanityFilter('en')
    filter_word = 'anal'
    unfilter_word = 'analysis'
    for text in [' ' + filter_word, ' ' + filter_word + ' ',
                 filter_word, filter_word + ' ' + unfilter_word]:
        assert profanity_filter.match(text) is True
    for text in [' ' + unfilter_word, unfilter_word, unfilter_word + ' ']:
        assert profanity_filter.match(text) is False


def test_sentence_normalizer():
    normalizer = MosesNormalizer('en')
    assert normalizer('    hello  world!!".\t\t\r') == ' hello world!!."  '
    assert normalizer(
        b'We therefore defend, and will continue to defend wherever necessary, our position of \xe2\x80\x98no diversion\xe2\x80\x99.\n'.decode('utf-8')) == \
           "We therefore defend, and will continue to defend wherever necessary, our position of 'no diversion'. "
    normalizer = MosesNormalizer('en', remove_non_printable_char=False)
    assert normalizer('    hello  world!!".\t\t\r') == ' hello world!!."\t\t'
    normalizer = MosesNormalizer('en', remove_non_printable_char=False, unicode_norm_form='NFKC')
    assert normalizer('    hello  world!!"⁵.\t\t\r') == ' hello world!!"5.\t\t'