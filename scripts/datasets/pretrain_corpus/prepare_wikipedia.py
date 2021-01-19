"""Prepare the Wikipedia dataset that contain cleaned articles of all languages."""
import os
import sys
import glob
import math
import time
import tarfile
import argparse
import multiprocessing
from collections import defaultdict
from gluonnlp.utils.misc import download, load_checksum_stats
from gluonnlp.utils.lazy_imports import try_import_wikiextractor
from gluonnlp.base import get_repo_url
from itertools import islice
import nltk
import statistics

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_TARGET_PATH = os.path.join(_CURR_DIR, '../../processing/')
sys.path.append(_TARGET_PATH)
from segment_sentences import Sharding, segment_sentences, NLTKSegmenter


_CITATION = """\
@ONLINE {wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}
"""

# See https://en.wikipedia.org/wiki/List_of_Wikipedias for details
__LANGUAGES_BANK = [
    "aa", "ab", "ace", "ady", "af", "ak", "als", "am", "an", "ang", "ar", "arc",
    "arz", "as", "ast", "atj", "av", "ay", "az", "azb", "ba", "bar", "bat-smg",
    "bcl", "be", "be-x-old", "bg", "bh", "bi", "bjn", "bm", "bn", "bo", "bpy",
    "br", "bs", "bug", "bxr", "ca", "cbk-zam", "cdo", "ce", "ceb", "ch", "cho",
    "chr", "chy", "ckb", "co", "cr", "crh", "cs", "csb", "cu", "cv", "cy", "da",
    "de", "din", "diq", "dsb", "dty", "dv", "dz", "ee", "el", "eml", "en", "eo",
    "es", "et", "eu", "ext", "fa", "ff", "fi", "fiu-vro", "fj", "fo", "fr",
    "frp", "frr", "fur", "fy", "ga", "gag", "gan", "gd", "gl", "glk", "gn",
    "gom", "gor", "got", "gu", "gv", "ha", "hak", "haw", "he", "hi", "hif",
    "ho", "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ig", "ii",
    "ik", "ilo", "inh", "io", "is", "it", "iu", "ja", "jam", "jbo", "jv", "ka",
    "kaa", "kab", "kbd", "kbp", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko",
    "koi", "krc", "ks", "ksh", "ku", "kv", "kw", "ky", "la", "lad", "lb",
    "lbe", "lez", "lfn", "lg", "li", "lij", "lmo", "ln", "lo", "lrc", "lt",
    "ltg", "lv", "mai", "map-bms", "mdf", "mg", "mh", "mhr", "mi", "min", "mk",
    "ml", "mn", "mr", "mrj", "ms", "mt", "mus", "mwl", "my", "myv", "mzn", "na",
    "nah", "nap", "nds", "nds-nl", "ne", "new", "ng", "nl", "nn", "no", "nov",
    "nrm", "nso", "nv", "ny", "oc", "olo", "om", "or", "os", "pa", "pag", "pam",
    "pap", "pcd", "pdc", "pfl", "pi", "pih", "pl", "pms", "pnb", "pnt", "ps",
    "pt", "qu", "rm", "rmy", "rn", "ro", "roa-rup", "roa-tara", "ru", "rue",
    "rw", "sa", "sah", "sat", "sc", "scn", "sco", "sd", "se", "sg", "sh", "si",
    "simple", "sk", "sl", "sm", "sn", "so", "sq", "sr", "srn", "ss", "st",
    "stq", "su", "sv", "sw", "szl", "ta", "tcy", "te", "tet", "tg", "th", "ti",
    "tk", "tl", "tn", "to", "tpi", "tr", "ts", "tt", "tum", "tw", "ty", "tyv",
    "udm", "ug", "uk", "ur", "uz", "ve", "vec", "vep", "vi", "vls", "vo", "wa",
    "war", "wo", "wuu", "xal", "xh", "xmf", "yi", "yo", "za", "zea", "zh",
    "zh-classical", "zh-min-nan", "zh-yue", "zu"]

_BASE_URL_TMPL\
    = "https://dumps.wikimedia.org/{lang}wiki/{date}/{lang}wiki-{date}-pages-articles.xml.bz2"
_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_URL_FILE_STATS_PATH = os.path.join(_CURR_DIR, '..', 'url_checksums', 'wikipedia.txt')
_URL_FILE_STATS = load_checksum_stats(_URL_FILE_STATS_PATH)
_URLS = {
    'wikipedia-en-20200620':
        get_repo_url() + 'pretrain_corpus/wikipedia-en-20200620.tar.gz',
}


def get_url(lang, date):
    return _BASE_URL_TMPL.format(lang=lang, date=date)



def get_formatting_list(wiki_path, recursive=False):
    """
    get formatting list of file names from extracted content
    """
    filenames = []
    for dirname in glob.glob(os.path.join(wiki_path, '*'), recursive=False):
        for filename in glob.glob(os.path.join(dirname, 'wiki_*'), recursive=recursive):
            filenames.append(filename)
    return filenames


def merge(x):
    """
    Puts one article per line
    """
    file_list, output_filename = x
    article_lines = []
    article_open = False

    with open(output_filename, mode='w', newline='\n') as ofile:
        for filename in file_list:
            with open(filename, mode='r', newline='\n') as file:
                for line in file:
                    if '<doc id=' in line:
                        article_open = True
                    elif '</doc>' in line:
                        article_open = False
                        for oline in article_lines[1:]:
                            if oline != '\n':
                                ofile.write(oline.rstrip() + " ")
                        ofile.write("\n\n")
                        article_lines = []
                    else:
                        if article_open:
                            article_lines.append(line)


def get_parser():
    parser = argparse.ArgumentParser(description='Download and Prepare the Wikipedia')
    parser.add_argument('--mode', type=str,
                        default='download+format',
                        choices=['download', 'format', 'download+format', 'download_prepared'],
                        help='Specify the action you want the app to take. '
                             '"download" means to download the Wikipedia dump. '
                             '"format" means to extract the content and '
                             'format it for pretraining. "download+format" means to combine '
                             'these two options'
                             '"download_prepared" downloads the prepared txt from S3 directly')
    parser.add_argument('--lang', type=str, default='en',
                        help='Language of the wikipedia dump file.'
                             'We only support English and Chinese for current version')
    parser.add_argument('--date', type=str, default='latest',
                        help='Date of the wikipedia dump file. You can choose a date like '
                             '"--date 20200201" or use "--date latest"')
    parser.add_argument("-i", "--input", default=None,
                        help="path to XML wiki dump file.")
    parser.add_argument("-o", "--output", default="wikicorpus",
                        help="directory for downloaded or formatted files")
    parser.add_argument("-b", "--bytes", default="100M",
                        help="maximum bytes per extracted file (default %(default)s)",
                        metavar="n[KMG]")
    parser.add_argument("--num_process", type=int, default=os.cpu_count(),
                        help="number of processes for multiprocessing")
    parser.add_argument("--num_out_files", type=int, default=1000,
                        help="Number of desired output files, where each is processed"
                             " independently by a worker.")
    parser.add_argument("--segment_sentences", action='store_true',
                        help="directory for downloaded  files")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress reporting progress info")
    parser.add_argument("--segment_num_worker", type=int, default=8,
                        help="process num when segmenting articles")
    return parser


def download_wikicorpus(lang, date, output):
    """
    lang: the language code such as en, zh
    date: string, the date of the Wikipedia with format of YYYYMMDD, or 'latest'.
    """
    if not os.path.exists(output):
        os.makedirs(output)
    if lang not in __LANGUAGES_BANK:
        raise ValueError('Unsupported language code')
    language = lang.replace('-', '_')
    output_file = os.path.join(output, 'download', language, date,
                               'wikicorpus.xml.bz2')
    download(get_url(language, date), output_file)
    return output_file


def format_wikicorpus(input, output, bytes, num_process, num_out_files, quiet):
    if input is None:
        raise ValueError('input file is empty.')
    if not input.endswith('xml.bz2'):
        raise ValueError('input file not *.xml.bz2.')
    if not os.path.exists(output):
        os.makedirs(output)

    # Use WikiExtractor to extract the content
    wikiextractor = try_import_wikiextractor()
    from wikiextractor import WikiExtractor
    wiki_path = os.path.join(output, 'extracted')
    # Overwrite the sys.argv
    sys.argv = ['prog', '-b', bytes, '-o', wiki_path, input]
    if quiet:
        sys.argv.append('--quiet')
    wikiextractor.WikiExtractor.main()

    # Merge extracted content into txt files
    prepared_path = os.path.join(output, 'prepared_wikipedia')
    if not os.path.exists(prepared_path):
        os.makedirs(prepared_path)
    filenames = get_formatting_list(wiki_path, recursive=True)
    num_files = len(filenames)
    num_out_files = min(num_out_files, num_files)
    file_volume = math.ceil(num_files / num_out_files)
    splited_files = [filenames[i: i + file_volume] for i in range(0, num_files, file_volume)]
    num_out_files = len(splited_files)
    output_files = [
        os.path.join(
            prepared_path,
            "wikipedia-prepared-{}.txt".format(
                str(i).zfill(4))) for i in range(num_out_files)]
    print("All prepared raw text will be saved in {} txt files".format(num_out_files))
    num_process = min(num_process, num_out_files)
    print('Start preprocessing {} text files with {} cores'.format(num_files, num_process))
    process_args = [(splited_files[i], output_files[i]) for i in range(num_out_files)]

    start_time = time.time()
    with multiprocessing.Pool(num_process) as pool:
        f_read = 0
        for i, _ in enumerate(pool.imap(merge, process_args)):
            elapsed = time.time() - start_time
            f_read += len(splited_files[i])
            print("prepared {:} files, Elapsed: {:.2f}s, ETA: {:.2f}s, ".format(
                f_read, elapsed, (num_files - f_read) / (num_files / elapsed)))
    print("Done preparation within {:.2f} seconds".format(elapsed))


def main(args):
    num_process = min(multiprocessing.cpu_count(), args.num_process)
    if args.mode == 'download':
        download_wikicorpus(args.lang, args.date, args.output)
    elif args.mode == 'format':
        format_wikicorpus(args.input, args.output, args.bytes, num_process,
                          args.num_out_files, args.quiet)
    elif args.mode == 'download+format':
        downloaded_file = download_wikicorpus(args.lang, args.date, args.output)
        format_wikicorpus(downloaded_file, args.output, args.bytes, num_process,
                          args.num_out_files, args.quiet)
    elif args.mode == 'download_prepared':
        url = _URLS['wikipedia-en-20200620']
        file_hash = _URL_FILE_STATS[url]
        target_download_location = os.path.join(args.output,
                                                os.path.basename(url))
        download(url, target_download_location, sha1_hash=file_hash)
        tar = tarfile.open(target_download_location)
        names = tar.getnames()
        print('Start unarchiving raw text files')
        start_time = time.time()
        for name in names:
            tar.extract(name, path=args.output)
        tar.close()
        print("Done unarchiving within {:.2f} seconds".format(time.time() - start_time))
    else:
        raise NotImplementedError
    if args.segment_sentences:
        print("start to transfer bookcorpus to one sentence per line")
        t1 = time.time()
        segmenter = NLTKSegmenter()
        original_name = os.path.join(args.output, 'prepared_wikipedia')
        output_name = os.path.join(args.output, 'one_sentence_per_line/')
        if not os.path.exists(output_name):
            os.mkdir(output_name)
        input_names = os.listdir(original_name)
        for i in range(len(input_names)):
            input_names[i]=os.path.join(original_name, input_names[i])
        sharding = Sharding(input_names, output_name, 256, 1, 0,
                            args.segment_num_worker)

        sharding.load_articles()
        sharding.segment_articles_into_sentences()
        t2 = time.time()
        print("transfer cost:{}".format(t2 - t1))


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
