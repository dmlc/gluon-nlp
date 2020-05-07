from typing import List, Union, IO, AnyStr, Tuple, Optional
import re
import os
import argparse
import zipfile
import shutil
import functools
import tarfile
import gzip
from xml.etree import ElementTree
from gluonnlp.data.filtering import ProfanityFilter
from gluonnlp.utils.misc import file_line_number, download
from gluonnlp.base import get_data_home_dir

# The datasets are provided by WMT2014-WMT2019 and can be freely used for research purposes.
# You will need to cite the WMT14-WMT19 shared task overview paper and additional citation
# requirements for specific individual datasets
#   http://www.statmt.org/wmt14/translation-task.html to
#   http://www.statmt.org/wmt19/translation-task.html


_CITATIONS = """
@inproceedings{ziemski2016united,
  title={The united nations parallel corpus v1. 0},
  author={Ziemski, Micha{\l} and Junczys-Dowmunt, Marcin and Pouliquen, Bruno},
  booktitle={Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)},
  pages={3530--3534},
  year={2016}
}

@inproceedings{barrault2019findings,
  title={Findings of the 2019 conference on machine translation (wmt19)},
  author={Barrault, Lo{\"\i}c and Bojar, Ond{\v{r}}ej and Costa-juss{\`a}, Marta R and Federmann, Christian and Fishel, Mark and Graham, Yvette and Haddow, Barry and Huck, Matthias and Koehn, Philipp and Malmasi, Shervin and others},
  booktitle={Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1)},
  pages={1--61},
  year={2019}
}
"""

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_BASE_DATASET_PATH = os.path.join(get_data_home_dir(), 'wmt')
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'wmt.txt')
_URL_FILE_STATS = dict()
for line in open(_URL_FILE_STATS_PATH, 'r', encoding='utf-8'):
    url, hex_hash, file_size = line.strip().split()
    _URL_FILE_STATS[url] = hex_hash


# Here, we will make sure that the languages follow the standard ISO 639-1 language tag.
# Also, for more information related to the language tag, you may refer to
# https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
_PARA_URLS = {
    'europarl': {
        'v7': {
            'cs-en': {
                'url': 'http://www.statmt.org/europarl/v7/cs-en.tgz',
                'cs': 'europarl-v7.cs-en.cs',
                'en': 'europarl-v7.cs-en.en',
            },
            'de-en': {
                'url': 'http://www.statmt.org/europarl/v7/de-en.tgz',
                'de': 'europarl-v7.de-en.de',
                'en': 'europarl-v7.de-en.en',
            }
        },
        'v8': {
            'url': 'http://data.statmt.org/wmt18/translation-task/training-parallel-ep-v8.tgz',
            'fi-en': {
                'fi': 'training/europarl-v8.fi-en.fi',
                'en': 'training/europarl-v8.fi-en.en'
            },
            'et-en': {
                'et': 'training/europarl-v8.et-en.et',
                'en': 'training/europarl-v8.et-en.en'
            }
        },
        'v9': {
            'cs-en': {
                'url': 'http://www.statmt.org/europarl/v9/training/europarl-v9.cs-en.tsv.gz',
                'all': 'europarl-v9.cs-en.tsv'
            },
            'de-en': {
                'url': 'http://www.statmt.org/europarl/v9/training/europarl-v9.de-en.tsv.gz',
                'all': 'europarl-v9.de-en.tsv'
            },
            'fi-en': {
                'url': 'http://www.statmt.org/europarl/v9/training/europarl-v9.fi-en.tsv.gz',
                'all': 'europarl-v9.fi-en.tsv'
            },
            'lt-en': {
                'url': 'http://www.statmt.org/europarl/v9/training/europarl-v9.lt-en.tsv.gz',
                'all': 'europarl-v9.lt-en.tsv'
            }
        }
    },
    'paracrawl': {
        'r3': {
            'en-cs': {
                'url': 'https://s3.amazonaws.com/web-language-models/paracrawl/release3/en-cs.bicleaner07.tmx.gz',
                'all': 'en-cs.bicleaner07.tmx'
            },
            'en-de': {
                'url': 'https://s3.amazonaws.com/web-language-models/paracrawl/release3/en-de.bicleaner07.tmx.gz',
                'all': 'en-de.bicleaner07.tmx'
            },
            'en-fi': {
                'url': 'https://s3.amazonaws.com/web-language-models/paracrawl/release3/en-fi.bicleaner07.tmx.gz',
                'all': 'en-fi.bicleaner07.tmx'
            },
            'en-lt': {
                'url': 'https://s3.amazonaws.com/web-language-models/paracrawl/release3/en-lt.bicleaner07.tmx.gz',
                'all': 'en-lt.bicleaner07.tmx'
            }
        }
    },
    'commoncrawl': {
        'wmt13': {
            'url': 'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
            'de-en': {
                'de': 'commoncrawl.de-en.de',
                'en': 'commoncrawl.de-en.en',
            }
        }
    },
    'newscommentary': {
        'v9': {
            'url': 'http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz',
            'de-en': {
                'de': 'training/news-commentary-v9.de-en.de',
                'en': 'training/news-commentary-v9.de-en.en'
            }
        },
        'v10': {
            'url': 'http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz',
            'de-en': {
                'de': 'news-commentary-v10.de-en.de',
                'en': 'news-commentary-v10.de-en.de'
            }
        },
        'v11': {
            'url': 'http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz',
            'de-en': {
                'de': 'training-parallel-nc-v11/news-commentary-v11.de-en.de',
                'en': 'training-parallel-nc-v11/news-commentary-v11.de-en.en'
            }
        },
        'v12': {
            'url': 'http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz',
            'de-en': {
                'de': 'training/news-commentary-v12.de-en.de',
                'en': 'training/news-commentary-v12.de-en.en',
            },
            'zh-en': {
                'zh': 'training/news-commentary-v12.zh-en.zh',
                'en': 'training/news-commentary-v12.zh-en.en'
            }
        },
        'v13': {
            'url': 'http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz',
            'de-en': {
                'de': 'training-parallel-nc-v13/news-commentary-v13.de-en.de',
                'en': 'training-parallel-nc-v13/news-commentary-v13.de-en.en',
            },
            'zh-en': {
                'zh': 'training-parallel-nc-v13/news-commentary-v13.zh-en.zh',
                'en': 'training-parallel-nc-v13/news-commentary-v13.zh-en.en'
            }
        },
        'v14': {
            'de-en': {
                'url': 'http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz',
                'all': 'news-commentary-v14.de-en.tsv'
            },
            'en-zh': {
                'url': 'http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-zh.tsv.gz',
                'all': 'news-commentary-v14.en-zh.tsv'
            }
        }
    },
    'wikititles': {
        'v1': {
            'cs-en': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.cs-en.tsv.gz',
                'all': 'wikititles-v1.cs-en.tsv'
            },
            'cs-pl': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.cs-pl.tsv.gz',
                'all': 'wikititles-v1.cs-pl.tsv'
            },
            'de-en': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.de-en.tsv.gz',
                'all': 'wikititles-v1.de-en.tsv'
            },
            'es-pt': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.es-pt.tsv.gz',
                'all': 'wikititles-v1.es-pt.tsv'
            },
            'fi-en': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.fi-en.tsv.gz',
                'all': 'wikititles-v1.fi-en.tsv'
            },
            'gu-en': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.gu-en.tsv.gz',
                'all': 'wikititles-v1.gu-en.tsv'
            },
            'hi-ne': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.hi-ne.tsv.gz',
                'all': 'wikititles-v1.hi-ne.tsv'
            },
            'kk-en': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.kk-en.tsv.gz',
                'all': 'wikititles-v1.kk-en.tsv'
            },
            'lt-en': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.lt-en.tsv.gz',
                'all': 'wikititles-v1.lt-en.tsv'
            },
            'ru-en': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.ru-en.tsv.gz',
                'all': 'wikititles-v1.ru-en.tsv'
            },
            'zh-en': {
                'url': 'http://data.statmt.org/wikititles/v1/wikititles-v1.zh-en.tsv.gz',
                'all': 'wikititles-v1.zh-en.tsv'
            }
        }
    },
    'uncorpus': {
        'v1': {
            'en-zh': {
                'url': ['https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-zh.tar.gz.00',
                        'https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-zh.tar.gz.01'],
                'en': 'en-zh/UNv1.0.en-zh.en',
                'zh': 'en-zh/UNv1.0.en-zh.zh'
            }
        }
    },
    # For the CWMT dataset, you can also download them from the official location: http://nlp.nju.edu.cn/cwmt-wmt/
    # Currently, this version is processed via https://gist.github.com/sxjscience/54bedd68ce3fb69b3b1b264377efb5a5
    'cwmt': {
        'url': 'https://gluonnlp-numpy-data.s3-us-west-2.amazonaws.com/wmt/cwmt.tar.gz',
        'zh-en': {
            'en': 'cwmt/cwmt-zh-en.en',
            'zh': 'cwmt/cwmt-zh-en.zh'
        }
    },
    'rapid': {
        '2016': {
            'url': 'http://data.statmt.org/wmt17/translation-task/rapid2016.tgz',
            'de-en': {
                'de': 'rapid2016.de-en.de',
                'en': 'rapid2016.de-en.en'
            }
        },
        '2019': {
            'de-en': {
                'url': 'https://s3-eu-west-1.amazonaws.com/tilde-model/rapid2019.de-en.zip',
                'de': 'rapid2019.de-en.de',
                'en': 'rapid2019.de-en.en'
            }
        }
    },
}

_MONOLINGUAL_URLS = {
    'newscrawl': {
        '2007': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2007.de.shuffled.deduped.gz',
                'de': 'newscrawl2007.de',
            }
        },
        '2008': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2008.de.shuffled.deduped.gz',
                'de': 'newscrawl2008.de',
            }
        },
        '2009': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2009.de.shuffled.deduped.gz',
                'de': 'newscrawl2009.de',
            }
        },
        '20010': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2010.de.shuffled.deduped.gz',
                'de': 'newscrawl2010.de',
            }
        },
        '2011': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2011.de.shuffled.deduped.gz',
                'de': 'newscrawl2011.de',
            }
        },
        '2012': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2012.de.shuffled.deduped.gz',
                'de': 'newscrawl2012.de',
            }
        },
        '2013': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2013.de.shuffled.deduped.gz',
                'de': 'newscrawl2013.de',
            }
        },
        '2014': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2014.de.shuffled.deduped.gz',
                'de': 'newscrawl2014.de',
            }
        },
        '2015': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2015.de.shuffled.deduped.gz',
                'de': 'newscrawl2015.de',
            }
        },
        '2016': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2016.de.shuffled.deduped.gz',
                'de': 'newscrawl2016.de',
            }
        },
        '2017': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2017.de.shuffled.deduped.gz',
                'de': 'newscrawl2017.de',
            }
        },
        '2018': {
            'de': {
                'url': 'http://data.statmt.org/news-crawl/de/news.2018.de.shuffled.deduped.gz',
                'de': 'newscrawl2018.de',
            }
        },
    }
}


def _clean_space(s: str):
    """Removes trailing and leading spaces and collapses multiple consecutive internal spaces to a single one.
    This is borrowed from sacrebleu: https://github.com/mjpost/sacreBLEU/blob/069b0c88fceb29f3e24c3c19ba25342a3e7f96cb/sacrebleu.py#L1077

    Parameters
    ----------
    s
        The input string

    Returns
    -------
    ret
        The cleaned string
    """
    return re.sub(r'\s+', ' ', s.strip())


def _get_buffer(path_or_buffer: Union[str, IO[AnyStr]], mode='r'):
    if isinstance(path_or_buffer, str):
        buf = open(path_or_buffer, mode)
    else:
        buf = path_or_buffer
    return buf


def parse_sgm(path_or_buffer: Union[str, IO[AnyStr]],
              out_path_or_buffer: Optional[Union[str, IO[AnyStr]]] = None,
              return_sentences=False,
              clean_space=True) -> Optional[List[str]]:
    """Returns sentences from a single SGML file. This is compatible to the behavior of
    `input-from-sgm.perl` in
    https://github.com/moses-smt/mosesdecoder/blob/a89691fee395bb7eb6dfd51e368825f0578f437d/scripts/ems/support/input-from-sgm.perl

    Parameters
    ----------
    path_or_buffer
        The source path to parse the file
    out_path_or_buffer
        The output path
    return_sentences
        Whether to return the parsed sentences
    clean_space
        Whether to clean the spaces in the sentence with the similar strategy in
         input-from-sgm.perl.

    Returns
    -------
    sentences
        The list contains the parsed sentences in the input file.
        If the return_sentences is False, return None.
    """
    if out_path_or_buffer is None:
        assert return_sentences, 'Must return sentences if the output path is not specified!'
    if return_sentences:
        sentences = []
    else:
        sentences = None
    f_buffer = _get_buffer(path_or_buffer, 'r')
    of_buffer = _get_buffer(out_path_or_buffer, 'w')
    seg_re = re.compile(r'<seg.*?>(.*)</seg>.*?')
    for line in f_buffer:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        seg_match = re.match(seg_re, line)
        if seg_match:
            assert len(seg_match.groups()) == 1,\
                'File content is not supported, unmatched line: {}'.format(line)
            line = seg_match.groups()[0]
            if clean_space:
                line = _clean_space(line)
            if of_buffer is not None:
                of_buffer.write(line + '\n')
            if sentences is not None:
                sentences.append(line)
    if of_buffer is not None:
        of_buffer.close()
    return sentences


def parse_paracrawl_tmx(path_or_buffer, src_lang, tgt_lang, out_src_path, out_tgt_path,
                        clean_space=True, filter_profanity=False):
    candidate_lang = {src_lang, tgt_lang}
    sent_num = 0
    if filter_profanity:
        src_profanity_filter = ProfanityFilter(langs=[src_lang])
        tgt_profanity_filter = ProfanityFilter(langs=[tgt_lang])
    has_src = False
    has_tgt = False
    src_sentence = None
    tgt_sentence = None
    f = _get_buffer(path_or_buffer)
    src_out_f = open(out_src_path, 'w', encoding='utf-8')
    tgt_out_f = open(out_tgt_path, 'w', encoding='utf-8')
    for i, (_, elem) in enumerate(ElementTree.iterparse(f)):
        if elem.tag == "tu":
            for tuv in elem.iterfind("tuv"):
                lang = None
                for k, v in tuv.items():
                    if k.endswith('}lang'):
                        assert v in candidate_lang,\
                            'Find language={} in data, which is not the same as either' \
                            ' the source/target languages={}/{}'.format(v, src_lang, tgt_lang)
                        lang = v
                        break
                if lang is not None:
                    segs = tuv.findall("seg")
                    assert len(segs) == 1, "Invalid number of segments: {}".format(len(segs))
                    if lang == src_lang:
                        assert not has_src
                        has_src = True
                        src_sentence = segs[0].text
                    else:
                        assert not has_tgt
                        has_tgt = True
                        tgt_sentence = segs[0].text
                    if has_src and has_tgt:
                        has_src, has_tgt = False, False
                        if clean_space:
                            # Merge the spaces
                            src_sentence = _clean_space(src_sentence)
                            tgt_sentence = _clean_space(tgt_sentence)
                        if filter_profanity:
                            if src_profanity_filter.match(src_sentence)\
                                    or tgt_profanity_filter.match(tgt_sentence):
                                continue
                        sent_num += 1
                        if sent_num % 500000 == 0:
                            print('Processed {} sentences'.format(sent_num))
                        src_out_f.write(src_sentence + '\n')
                        tgt_out_f.write(tgt_sentence + '\n')
            elem.clear()
    src_out_f.close()
    tgt_out_f.close()
    assert has_src or has_tgt,\
        'The number of source and target sentences are not the same.'


def parse_tsv(path_or_buffer, src_out_path, tgt_out_path):
    in_f = _get_buffer(path_or_buffer, 'r')
    src_out_f = _get_buffer(src_out_path, 'w')
    tgt_out_f = _get_buffer(tgt_out_path, 'w')
    for line in in_f:
        line = line.strip()
        split_data = line.split('\t')
        if len(split_data) == 2:
            # Here, some lines may be corrupted and may not have a target translation
            src_sentence, tgt_sentence = split_data
            src_out_f.write(src_sentence + '\n')
            tgt_out_f.write(tgt_sentence + '\n')


def split_lang_pair(pair: str = 'de-en') -> Tuple[str, str]:
    try:
        src_lang, tgt_lang = pair.split('-')
    except ValueError:
        raise ValueError('pair must be format like "en-de", "zh-en". Received {}'
                         .format(pair))
    return src_lang, tgt_lang


def concatenate_files(fname_l: List[str],
                      out_fname: Optional[str] = None,
                      chunk_size: int = 128 * 1024) -> str:
    """Concatenate multiple files into a single file. This is used to recover a large file that has
    been split into multiple parts. E.g.,

    UNv1.0.en-zh.tar.gz.00, UNv1.0.en-zh.tar.gz.01 --> UNv1.0.en-zh.tar.gz

    Parameters
    ----------
    fname_l
    out_fname
    chunk_size

    Returns
    -------
    ret
    """
    assert len(fname_l) > 1
    ext_l = []
    base_prefix, ext = os.path.splitext(fname_l[0])
    ext_l.append(ext)
    for i in range(1, len(fname_l)):
        prefix, ext = os.path.splitext(fname_l[i])
        ext_l.append(ext)
        if prefix != base_prefix:
            raise ValueError('Cannot concatenate the input files! The prefix does not match! '
                             'Find prefix={}, Expected prefix={}'.format(prefix, base_prefix))
    fname_ext_l = sorted(zip(fname_l, ext_l), key=lambda ele: ele[1])
    if out_fname is None:
        out_fname = base_prefix
    with open(out_fname, 'wb') as of:
        for fname, _ in fname_ext_l:
            with open(fname, 'rb') as infile:
                for block in iter(functools.partial(infile.read, chunk_size), b''):
                    of.write(block)
    return out_fname


def extract_mono_corpus(compressed_data_path, src_lang, src_name, out_src_path):
    tmp_dir = os.path.join(os.path.dirname(compressed_data_path), 'raw_data')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # Uncompress data
    if compressed_data_path.endswith('.gz'):
        with gzip.open(compressed_data_path) as f_in:
            with open(os.path.join(tmp_dir, src_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise NotImplementedError('Cannot process {}'.format(compressed_data_path))
    # Parse data and move to the required src/tgt path
    
    shutil.copyfile(os.path.join(tmp_dir, src_name), out_src_path)
        
    # Clean-up
    shutil.rmtree(tmp_dir)


def fetch_mono_dataset(selection: Union[str, List[str], List[List[str]]],
                       lang: str = 'de',
                       path: Optional[str] = _BASE_DATASET_PATH,
                       overwrite: bool = False) -> List[str]:
    """Fetch the monolingual dataset provided by WMT

    Parameters
    ----------
    selection
    lang
    path
    overwrite

    Returns
    -------
    src_corpus_paths
    """
    base_url_info = _MONOLINGUAL_URLS
    if isinstance(selection, str):
        selection = [selection]
    elif isinstance(selection, list):
        if isinstance(selection[0], list):
            corpus_paths = []
            for ele in selection:
                ele_corpus_paths =\
                    fetch_mono_dataset(ele, lang, path, overwrite)
                corpus_paths.extend(ele_corpus_paths)
            return corpus_paths
    else:
        raise NotImplementedError
    for sel in selection:
        base_url_info = base_url_info[sel]

    # Check the pair is valid
    available_lang = set(base_url_info.keys())
    if 'url' in available_lang:
        available_lang.remove('url')
    if lang in available_lang:
        matched_lang = '{}'.format(lang)
    else:
        raise ValueError('Unsupported lang, lang={}. All supported: {}'
                         .format(lang, available_lang))
    save_dir_path = os.path.join(path, *(selection + [matched_lang]))
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    out_path = os.path.join(save_dir_path, lang + '.txt')
    # Check for whether we can load the cached version
    # TODO we can do something smarter here
    if os.path.exists(out_path) and not overwrite:
        print('Found data in {}, skip:\n'
              '\tSource: {}\n'.format(selection + [lang], out_path))
        return [out_path]
    lang_data_info = base_url_info[matched_lang]
    if 'url' in lang_data_info:
        url_l = lang_data_info['url']
    else:
        url_l = base_url_info['url']
    # Download the data + Concatenate the file-parts (if necessary)
    download_fname_l = []
    if isinstance(url_l, str):
        url_l = [url_l]
    for url in url_l:
        original_filename = url[url.rfind("/") + 1:]
        sha1_hash = _URL_FILE_STATS[url]
        if 'url' in lang_data_info:
            save_path_l = [path] + selection + [matched_lang, original_filename]
        else:
            save_path_l = [path] + selection + [original_filename]
        download_fname = download(url, path=os.path.join(*save_path_l), sha1_hash=sha1_hash)
        download_fname_l.append(download_fname)
    if len(download_fname_l) > 1:
        data_path = concatenate_files(download_fname_l)
    else:
        data_path = download_fname_l[0]
    
    src_name = lang_data_info[lang]
    print('Prepare data for {}\n'
          '\tCompressed File: {}\n'
          '\t{}: {}\n'.format(selection + [lang],
                            data_path,
                            lang, out_path))
    extract_mono_corpus(data_path,
                        src_lang=lang,
                        src_name=src_name,
                        out_src_path=out_path)
    return [out_path]


def extract_src_tgt_corpus(compressed_data_path,
                           data_lang_pair, src_lang, tgt_lang,
                           src_name, tgt_name, src_tgt_name,
                           out_src_path, out_tgt_path):
    data_src_lang, data_tgt_lang = split_lang_pair(data_lang_pair)
    if not ((src_lang == data_src_lang and tgt_lang == data_tgt_lang) or
            (src_lang == data_tgt_lang and tgt_lang == data_src_lang)):
        raise ValueError('Mismatch src/tgt language. Required pair={}, Given src={}, tgt={}'
                         .format(data_lang_pair, src_lang, tgt_lang))
    reverse_pair = (src_lang == data_tgt_lang) and (tgt_lang == data_src_lang)
    if src_tgt_name is not None:
        assert src_name is None and tgt_name is None
    tmp_dir = os.path.join(os.path.dirname(compressed_data_path), 'raw_data')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # Uncompress data
    if compressed_data_path.endswith('.tar.gz') or compressed_data_path.endswith('.tgz'):
        with tarfile.open(compressed_data_path) as f:
            if src_tgt_name is None:
                f.extract(src_name, tmp_dir)
                f.extract(tgt_name, tmp_dir)
            else:
                f.extract(src_tgt_name, os.path.join(tmp_dir, src_tgt_name))
    elif compressed_data_path.endswith('.gz'):
        assert src_tgt_name is not None
        with gzip.open(compressed_data_path) as f_in:
            with open(os.path.join(tmp_dir, src_tgt_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif compressed_data_path.endswith('.zip'):
        with zipfile.ZipFile(compressed_data_path) as zip_handler:
            if src_tgt_name is None:
                with zip_handler.open(src_name) as f_in:
                    with open(os.path.join(tmp_dir, src_name), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                with zip_handler.open(tgt_name) as f_in:
                    with open(os.path.join(tmp_dir, tgt_name), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                with zip_handler.open(src_tgt_name) as f_in:
                    with open(os.path.join(tmp_dir, src_tgt_name), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
    else:
        raise NotImplementedError('Cannot process {}'.format(compressed_data_path))
    # Parse data and move to the required src/tgt path
    if src_tgt_name is None:
        if src_name.endswith('.sgm'):
            parse_sgm(os.path.join(tmp_dir, src_name), out_src_path)
            parse_sgm(os.path.join(tmp_dir, tgt_name), out_tgt_path)
        else:
            shutil.copyfile(os.path.join(tmp_dir, src_name), out_src_path)
            shutil.copyfile(os.path.join(tmp_dir, tgt_name), out_tgt_path)
    else:
        if src_tgt_name.endswith('.tmx'):
            parse_paracrawl_tmx(os.path.join(tmp_dir, src_tgt_name),
                                src_lang=src_lang,
                                tgt_lang=tgt_lang,
                                out_src_path=out_src_path,
                                out_tgt_path=out_tgt_path,
                                clean_space=True,
                                filter_profanity=False)
        elif src_tgt_name.endswith('.tsv'):
            if reverse_pair:
                parse_tsv(os.path.join(tmp_dir, src_tgt_name), out_tgt_path, out_src_path)
            else:
                parse_tsv(os.path.join(tmp_dir, src_tgt_name), out_src_path, out_tgt_path)
        else:
            raise NotImplementedError
    # Clean-up
    shutil.rmtree(tmp_dir)


def fetch_wmt_parallel_dataset(selection: Union[str, List[str], List[List[str]]],
                               lang_pair: str = 'de-en',
                               path: Optional[str] = _BASE_DATASET_PATH,
                               overwrite: bool = False) -> Tuple[List[str], List[str]]:
    """

    Parameters
    ----------
    selection
    lang_pair
    path
    overwrite

    Returns
    -------
    src_corpus_paths
    target_corpus_paths
    """
    src_lang, tgt_lang = split_lang_pair(lang_pair)
    base_url_info = _PARA_URLS
    if isinstance(selection, str):
        selection = [selection]
    elif isinstance(selection, list):
        if isinstance(selection[0], list):
            src_corpus_paths = []
            tgt_corpus_paths = []
            for ele in selection:
                ele_src_corpus_paths, ele_tgt_corpus_paths =\
                    fetch_wmt_parallel_dataset(ele, lang_pair, path, overwrite)
                src_corpus_paths.extend(ele_src_corpus_paths)
                tgt_corpus_paths.extend(ele_tgt_corpus_paths)
            return src_corpus_paths, tgt_corpus_paths
    else:
        raise NotImplementedError
    for sel in selection:
        base_url_info = base_url_info[sel]
    # Check the pair is valid
    available_pairs = set(base_url_info.keys())
    if 'url' in available_pairs:
        available_pairs.remove('url')
    if str(src_lang) + '-' + str(tgt_lang) in available_pairs:
        matched_pair = '{}-{}'.format(src_lang, tgt_lang)
    elif str(tgt_lang) + '-' + str(src_lang) in available_pairs:
        matched_pair = '{}-{}'.format(tgt_lang, src_lang)
    else:
        raise ValueError('Unsupported pairs, src_lang={}, tgt_lang={}. All supported: {}'
                         .format(src_lang, tgt_lang, available_pairs))
    save_dir_path = os.path.join(path, *(selection + [matched_pair]))
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    out_src_path = os.path.join(save_dir_path, src_lang + '.txt')
    out_tgt_path = os.path.join(save_dir_path, tgt_lang + '.txt')
    # Check for whether we can load the cached version
    # TODO we can do something smarter here
    if os.path.exists(out_src_path) and os.path.exists(out_tgt_path) and not overwrite:
        print('Found data in {}, skip:\n'
              '\tSource: {}\n'
              '\tTarget: {}\n'.format(selection + [lang_pair], out_src_path, out_tgt_path))
        return [out_src_path], [out_tgt_path]
    pair_data_info = base_url_info[matched_pair]
    if 'url' in pair_data_info:
        url_l = pair_data_info['url']
    else:
        url_l = base_url_info['url']
    # Download the data + Concatenate the file-parts (if necessary)
    download_fname_l = []
    if isinstance(url_l, str):
        url_l = [url_l]
    for url in url_l:
        original_filename = url[url.rfind("/") + 1:]
        sha1_hash = _URL_FILE_STATS[url]
        if 'url' in pair_data_info:
            save_path_l = [path] + selection + [matched_pair, original_filename]
        else:
            save_path_l = [path] + selection + [original_filename]
        download_fname = download(url, path=os.path.join(*save_path_l), sha1_hash=sha1_hash)
        download_fname_l.append(download_fname)
    if len(download_fname_l) > 1:
        data_path = concatenate_files(download_fname_l)
    else:
        data_path = download_fname_l[0]
    if 'all' in pair_data_info:
        src_name, tgt_name, src_tgt_name = None, None, pair_data_info['all']
    else:
        src_name, tgt_name, src_tgt_name = pair_data_info[src_lang], pair_data_info[tgt_lang], None
    print('Prepare data for {}\n'
          '\tCompressed File: {}\n'
          '\t{}: {}\n'
          '\t{}: {}\n'.format(selection + [lang_pair],
                            data_path,
                            src_lang, out_src_path,
                            tgt_lang, out_tgt_path))
    extract_src_tgt_corpus(data_path,
                           data_lang_pair=matched_pair,
                           src_lang=src_lang,
                           tgt_lang=tgt_lang,
                           src_name=src_name,
                           tgt_name=tgt_name,
                           src_tgt_name=src_tgt_name,
                           out_src_path=out_src_path,
                           out_tgt_path=out_tgt_path)
    assert file_line_number(out_src_path) == file_line_number(out_tgt_path)
    return [out_src_path], [out_tgt_path]


def download_mono_train(lang: str = 'de', path: str = _BASE_DATASET_PATH)\
        -> List[str]:
    """Download the train dataset used for WMT2014

    Parameters
    ----------
    lang
    path

    Returns
    -------
    train_src_paths
    """
    if lang == 'de':
        train_src_paths =\
            fetch_mono_dataset([['newscrawl', '2017'],
                                ['newscrawl', '2018']],
                               lang=lang,
                               path=path)
    else:
        raise NotImplementedError
    return train_src_paths


def download_wmt14_train(lang_pair: str = 'en-de', path: str = _BASE_DATASET_PATH)\
        -> Tuple[List[str], List[str]]:
    """Download the train dataset used for WMT2014

    Parameters
    ----------
    lang_pair
    path

    Returns
    -------
    train_src_paths
    train_tgt_paths
    """
    if lang_pair == 'en-de' or lang_pair == 'de-en':
        train_src_paths, train_tgt_paths =\
            fetch_wmt_parallel_dataset([['europarl', 'v7'],
                                        ['commoncrawl', 'wmt13'],
                                        ['newscommentary', 'v9']], lang_pair, path=path)
    else:
        raise NotImplementedError
    return train_src_paths, train_tgt_paths


def download_wmt16_train(lang_pair: str = 'en-de', path: str = _BASE_DATASET_PATH)\
        -> Tuple[List[str], List[str]]:
    """Download the train dataset used for WMT2016

    Parameters
    ----------
    lang_pair
    path

    Returns
    -------
    train_src_paths
    train_tgt_paths

    """
    if lang_pair == 'en-de' or lang_pair == 'de-en':
        train_src_paths, train_tgt_paths = \
            fetch_wmt_parallel_dataset([['europarl', 'v7'],
                                        ['commoncrawl', 'wmt13'],
                                        ['newscommentary', 'v11']], lang_pair, path=path)
    else:
        raise NotImplementedError
    return train_src_paths, train_tgt_paths


def download_wmt17_train(lang_pair: str = 'en-de', path: str = _BASE_DATASET_PATH)\
        -> Tuple[List[str], List[str]]:
    """Download the train dataset used for WMT2017

    Parameters
    ----------
    lang_pair
    path

    Returns
    -------
    train_src_paths
    train_tgt_paths

    """
    if lang_pair == 'en-de' or lang_pair == 'de-en':
        train_src_paths, train_tgt_paths = \
            fetch_wmt_parallel_dataset([['europarl', 'v7'],
                                        ['commoncrawl', 'wmt13'],
                                        ['newscommentary', 'v12'],
                                        ['rapid', '2016']], lang_pair, path=path)
    elif lang_pair == 'zh-en' or lang_pair == 'en-zh':
        train_src_paths, train_tgt_paths = \
            fetch_wmt_parallel_dataset([['newscommentary', 'v13'],
                                        ['uncorpus', 'v1'],
                                        ['cwmt']], lang_pair, path=path)
    else:
        raise NotImplementedError
    return train_src_paths, train_tgt_paths


def get_parser():
    parser = argparse.ArgumentParser(description='Downloading and Preprocessing WMT Datasets.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wmt2014', 'wmt2017', 'newscrawl'],
                        help='The dataset to use.')
    parser.add_argument('--mono', action='store_true',
                        help='Download monolingual dataset.')
    parser.add_argument('--mono_lang', type=str, default='de',
                        help='The monolingual language.')                  
    parser.add_argument('--lang-pair', type=str, default='en-de',
                        help='The pair of source language and target language separated by "-", '
                             'e.g. "en-de", "en-zh".')
    parser.add_argument('--mode', choices=['path_only',
                                           'raw',
                                           'prebuild'],
                        default='raw',
                        help='If the mode is "path_only",'
                             '    the script will only output the path of the raw corpus.'
                             'If mode is "raw", the script will concatenate all the related'
                             '    corpus and save to the folder.'
                             'If mode is "prebuild", the script will directly download the'
                             '    prebuild dataset, which is ready for training')
    parser.add_argument('--prebuild_name', default=None, type=str,
                        help='Name of the prebuild dataset.')
    parser.add_argument('--save-path', type=str, default='wmt_data',
                        help='The path to save the dataset.')
    parser.add_argument('--prefix', type=str, default='train.raw',
                        help='The prefix of the saved raw files.')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the ')
    parser.add_argument('--cache-path', type=str, default=_BASE_DATASET_PATH,
                        help='The path to cache the downloaded files.')
    return parser


def mono_main(args):
    lang = args.mono_lang
    if args.dataset.lower() == 'newscrawl':
        if lang == 'de':
            train_src_paths =\
                download_mono_train('de', args.cache_path)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if args.mode == 'path_only':
        print('Dataset: {}/{}'.format(args.dataset, args.mono_lang))
        print('Train Source:')
        for path in train_src_paths:
            print('\t{}'.format(path))
    elif args.mode == 'raw':
        assert args.save_path is not None
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        print('Save to {}'.format(args.save_path))
        raw_src_path = os.path.join(args.save_path, '{}.{}'.format(args.prefix, lang))
        if not os.path.exists(raw_src_path) or args.overwrite:
            with open(raw_src_path, 'wb') as out_f:
                for ele_path in train_src_paths:
                    with open(ele_path, 'rb') as in_f:
                        shutil.copyfileobj(in_f, out_f)
    elif args.mode == 'prebuild':
        #TODO Download and extract the prebuild dataset
        assert args.prebuild_name is not None, 'Must specify the prebuild_name if you are ' \
                                               'going to download the prebuild data'
    else:
        raise NotImplementedError


def main(args):
    if args.mono:
        mono_main(args)
    else:
        src_lang, tgt_lang = split_lang_pair(args.lang_pair)
        if args.dataset.lower() == 'wmt2014':
            if (src_lang, tgt_lang) in [('en', 'de'), ('de', 'en')]:
                train_src_paths, train_tgt_paths =\
                    download_wmt14_train(args.lang_pair, args.cache_path)
            else:
                raise NotImplementedError
        elif args.dataset.lower() == 'wmt2016':
            if (src_lang, tgt_lang) in [('en', 'de'), ('de', 'en')]:
                train_src_paths, train_tgt_paths =\
                    download_wmt16_train(args.lang_pair, args.cache_path)
            else:
                raise NotImplementedError
        elif args.dataset.lower() == 'wmt2017':
            if (src_lang, tgt_lang) in [('en', 'de'), ('de', 'en'),
                                        ('zh', 'en'), ('en', 'zh')]:
                train_src_paths, train_tgt_paths =\
                    download_wmt17_train(args.lang_pair, args.cache_path)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if args.mode == 'path_only':
            print('Dataset: {}/{}'.format(args.dataset, args.lang_pair))
            print('Train Source:')
            for path in train_src_paths:
                print('\t{}'.format(path))
            print('Train Target:')
            for path in train_tgt_paths:
                print('\t{}'.format(path))
        elif args.mode == 'raw':
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            print('Save to {}'.format(args.save_path))
            raw_src_path = os.path.join(args.save_path, '{}.{}'.format(args.prefix, src_lang))
            raw_tgt_path = os.path.join(args.save_path, '{}.{}'.format(args.prefix, tgt_lang))
            if not os.path.exists(raw_src_path) or args.overwrite:
                with open(raw_src_path, 'wb') as out_f:
                    for ele_path in train_src_paths:
                        with open(ele_path, 'rb') as in_f:
                            shutil.copyfileobj(in_f, out_f)
            if not os.path.exists(raw_tgt_path) or args.overwrite:
                with open(raw_tgt_path, 'wb') as out_f:
                    for ele_path in train_tgt_paths:
                        with open(ele_path, 'rb') as in_f:
                            shutil.copyfileobj(in_f, out_f)
            assert file_line_number(raw_src_path) == file_line_number(raw_tgt_path)
        elif args.mode == 'prebuild':
            #TODO Download and extract the prebuild dataset
            assert args.prebuild_name is not None, 'Must specify the prebuild_name if you are ' \
                                                'going to download the prebuild data'
        else:
            raise NotImplementedError


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()

