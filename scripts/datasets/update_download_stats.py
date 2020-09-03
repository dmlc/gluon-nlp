import hashlib
import requests
import time
import os
import copy
from collections import OrderedDict
from gluonnlp.cli.data.machine_translation.prepare_wmt\
    import _PARA_URLS as wmt_para_urls, _MONOLINGUAL_URLS as wmt_mono_urls
from gluonnlp.cli.data.question_answering.prepare_squad import _URLS as squad_urls
from gluonnlp.cli.data.question_answering.prepare_triviaqa import _URLS as triviaqa_url
from gluonnlp.cli.data.question_answering.prepare_hotpotqa import _URLS as hotpotqa_urls
from gluonnlp.cli.data.question_answering.prepare_searchqa import _URLS as searchqa_urls
from gluonnlp.cli.data.language_modeling.prepare_lm import _URLS as lm_urls
from gluonnlp.cli.data.music_generation.prepare_music_midi import _URLS as midi_urls
from gluonnlp.cli.data.pretrain_corpus.prepare_gutenberg import _URLS as gutenberg_urls
from gluonnlp.cli.data.general_nlp_benchmark.prepare_glue import SUPERGLUE_TASK2PATH as superglue_urls
from gluonnlp.cli.data.general_nlp_benchmark.prepare_glue import GLUE_TASK2PATH as glue_urls
from gluonnlp.cli.data.general_nlp_benchmark.prepare_text_classification import TASK2PATH as text_classification_urls


_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_CHECK_SUM_BASE = os.path.join(_CURR_DIR, 'url_checksums')


def get_hash_and_size(obj, retries=5, algorithm='sha1', cache=None, save_path=None,
                      verify_ssl=True):
    """Fetch sha1 hash of all urls in the input obj"""
    def _get_hash_and_size(obj, retries, algorithm, cache=None, save_path=None):
        if isinstance(obj, str):
            if obj.startswith('http://') or obj.startswith('https://'):
                url = obj
                hex_hash = None
                file_size = None
                if cache is not None and obj in cache:
                    return obj, cache[obj]
                while retries + 1 > 0:
                    # Disable pyling too broad Exception
                    # pylint: disable=W0703
                    try:
                        if algorithm == 'sha1':
                            m = hashlib.sha1()
                        elif algorithm == 'sha256':
                            m = hashlib.sha256()
                        elif algorithm == 'md5':
                            m = hashlib.md5()
                        else:
                            raise NotImplementedError
                        print('Calculating hash of the file downloaded from {}...'.format(url))
                        start = time.time()
                        r = requests.get(url, stream=True, verify=verify_ssl)
                        if r.status_code != 200:
                            raise RuntimeError('Failed downloading url {}'.format(url))
                        f_size = 0
                        for chunk in r.iter_content(chunk_size=10240):
                            if chunk:  # filter out keep-alive new chunks
                                m.update(chunk)
                                f_size += len(chunk)
                        hex_hash = m.hexdigest()
                        file_size = f_size
                        end = time.time()
                        print('{}={}, size={}, Time spent={}'.format(algorithm, hex_hash, file_size,
                                                                     end - start))
                        if cache is None:
                            cache = OrderedDict()
                        cache[url] = (hex_hash, file_size)
                        if save_path is not None:
                            with open(save_path, 'a', encoding='utf-8') as of:
                                of.write('{} {} {}\n'.format(url, hex_hash, file_size))
                        break
                    except Exception as e:
                        retries -= 1
                        if retries <= 0:
                            raise e
                        print('download failed due to {}, retrying, {} attempt{} left'
                              .format(repr(e), retries, 's' if retries > 1 else ''))
                return obj, (hex_hash, file_size)
            else:
                return obj
        elif isinstance(obj, tuple):
            return tuple((_get_hash_and_size(ele, retries, algorithm, cache, save_path)
                          for ele in obj))
        elif isinstance(obj, list):
            return [_get_hash_and_size(ele, retries, algorithm, cache, save_path) for ele in obj]
        elif isinstance(obj, dict):
            return {k: _get_hash_and_size(v, retries, algorithm, cache, save_path)
                    for k, v in obj.items()}
        else:
            return obj
    if cache is None:
        cache = OrderedDict()
    else:
        cache = copy.deepcopy(cache)
    if save_path is not None and os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                url, hex_hash, file_size = line.split()
                cache[url] = (hex_hash, file_size)
    _get_hash_and_size(obj, retries, algorithm, cache, save_path)
    return obj, cache


if __name__ == '__main__':
    get_hash_and_size([wmt_para_urls, wmt_mono_urls],
                      save_path=os.path.join(_CHECK_SUM_BASE, 'wmt.txt'))
    get_hash_and_size(squad_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'squad.txt'))
    get_hash_and_size(hotpotqa_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'hotpotqa.txt'))
    get_hash_and_size(triviaqa_url,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'triviaqa.txt'))
    get_hash_and_size(gutenberg_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'gutenberg.txt'))
    get_hash_and_size(searchqa_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'searchqa.txt'))
    get_hash_and_size(lm_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'language_model.txt'))
    get_hash_and_size(midi_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'music_midi.txt'))
    get_hash_and_size(glue_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'glue.txt'))
    get_hash_and_size(superglue_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'superglue.txt'))
    get_hash_and_size(text_classification_urls,
                      save_path=os.path.join(_CHECK_SUM_BASE, 'text_classification.txt'))
