__all__ = ['glob', 'file_line_number', 'md5sum', 'sha1sum', 'naming_convention',
           'logging_config', 'set_seed', 'sizeof_fmt', 'grouper', 'repeat',
           'parse_ctx', 'load_checksum_stats', 'download', 'check_version',
           'init_comm', 'get_mxnet_visible_ctx', 'logerror', 'BooleanOptionalAction']

import argparse
import os
import sys
import inspect
import logging
import warnings
import functools
import uuid
from types import ModuleType
from typing import Optional
import numpy as np
import hashlib
import requests
import itertools
import random
try:
    import tqdm
except ImportError:
    tqdm = None
from .lazy_imports import try_import_boto3
from mxnet.gluon.utils import replace_file
import glob as _glob


S3_PREFIX = 's3://'


def glob(url, separator=','):
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards.
    Input may also include multiple patterns, separated by separator.

    Parameters
    ----------
    url : str
        The name of the files
    separator : str, default is ','
        The separator in url to allow multiple patterns in the input
    """
    patterns = [url] if separator is None else url.split(separator)
    result = []
    for pattern in patterns:
        result.extend(_glob.glob(os.path.expanduser(pattern.strip())))
    return result


def file_line_number(path: str) -> int:
    """

    Parameters
    ----------
    path
        The path to calculate the number of lines in a file.

    Returns
    -------
    ret
        The number of lines
    """
    ret = 0
    with open(path, 'rb') as f:
        for _ in f:
            ret += 1
        return ret


def md5sum(filename):
    """Calculate the md5sum of a file

    Parameters
    ----------
    filename
        Name of the file

    Returns
    -------
    ret
        The md5sum
    """
    with open(filename, mode='rb') as f:
        d = hashlib.md5()
        for buf in iter(functools.partial(f.read, 1024*100), b''):
            d.update(buf)
    return d.hexdigest()


def sha1sum(filename):
    """Calculate the sha1sum of a file

    Parameters
    ----------
    filename
        Name of the file

    Returns
    -------
    ret
        The sha1sum
    """
    with open(filename, mode='rb') as f:
        d = hashlib.sha1()
        for buf in iter(functools.partial(f.read, 1024*100), b''):
            d.update(buf)
    return d.hexdigest()


def naming_convention(file_dir, file_name):
    """Rename files with 8-character hash"""
    long_hash = sha1sum(os.path.join(file_dir, file_name))
    file_prefix, file_sufix = file_name.split('.')
    new_name = '{file_prefix}-{short_hash}.{file_sufix}'.format(
        file_prefix=file_prefix,
        short_hash=long_hash[:8],
        file_sufix=file_sufix)
    return new_name, long_hash


def logging_config(folder: Optional[str] = None,
                   name: Optional[str] = None,
                   logger: logging.Logger = logging.root,
                   level: int = logging.INFO,
                   console_level: int = logging.INFO,
                   console: bool = True,
                   overwrite_handler: bool = False) -> str:
    """Config the logging module. It will set the logger to save to the specified file path.

    Parameters
    ----------
    folder
        The folder to save the log
    name
        Name of the saved
    logger
        The logger
    level
        Logging level
    console_level
        Logging level of the console log
    console
        Whether to also log to console
    overwrite_handler
        Whether to overwrite the existing handlers in the logger

    Returns
    -------
    folder
        The folder to save the log file.
    """
    if name is None:
        name = inspect.stack()[-1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    need_file_handler = True
    need_console_handler = True
    # Check all loggers.
    if overwrite_handler:
        logger.handlers = []
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                need_console_handler = False
    logpath = os.path.join(folder, name + ".log")
    print("All Logs will be saved to {}".format(logpath))
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if need_file_handler:
        logfile = logging.FileHandler(logpath)
        logfile.setLevel(level)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)
    if console and need_console_handler:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logger.addHandler(logconsole)
    return folder


def logerror(logger: logging.Logger = logging.root):
    """A decorator that wraps the passed in function and logs exceptions.

    Parameters
    ----------
    logger: logging.Logger
        The logger to which to log the error.
    """
    def log_wrapper(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                # log the exception
                logger.exception(
                    f'{function.__name__}(args={args}, kwargs={kwargs}) failed:\n{e}.')
                raise e
        return wrapper
    return log_wrapper


def set_seed(seed):
    import mxnet as mx
    mx.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return '{:.1f} {}{}'.format(num, unit, suffix)
        num /= 1024.0
    return '{:.1f} {}{}'.format(num, 'Yi', suffix)


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def repeat(iterable, count=None):
    """Repeat a basic iterator for multiple rounds

    Parameters
    ----------
    iterable
        The basic iterable
    count
        Repeat the basic iterable for "count" times. If it is None, it will be an infinite iterator.

    Returns
    -------
    new_iterable
        A new iterable in which the basic iterator has been repeated for multiple rounds.
    """
    if count is None:
        while True:
            for sample in iterable:
                yield sample
    else:
        for i in range(count):
            for sample in iterable:
                yield sample


def parse_ctx(data_str):
    import mxnet as mx
    if data_str == '-1' or data_str == '':
        ctx_l = [mx.cpu()]
    else:
        ctx_l = [mx.gpu(int(x)) for x in data_str.split(',')]
    return ctx_l


def load_checksum_stats(path: str) -> dict:
    """

    Parameters
    ----------
    path
        Path to the stored checksum

    Returns
    -------
    file_stats
    """
    file_stats = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name, hex_hash, file_size = line.strip().split()
            file_stats[name] = hex_hash
            if name[8:27] == 'gluonnlp-numpy-data':
                new_name = name.replace('https://gluonnlp-numpy-data.s3-accelerate.amazonaws.com', 's3://gluonnlp-numpy-data')
                file_stats[new_name] = hex_hash

    return file_stats


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.

    Based on: https://github.com/ndrplz/google-drive-downloader
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, showsize=False):
        """Downloads a shared file from google drive into a given folder.
        Optionally unzips it.

        Parameters
        ----------
        file_id: str
            the file identifier.
            You can obtain it from the sharable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        showsize: bool
            optional, if True print the current download size.
        """

        destination_directory = os.path.dirname(dest_path)
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        if not os.path.exists(dest_path) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_id, dest_path), end='')
            sys.stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL,
                                   params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL,
                                       params=params, stream=True)

            if showsize:
                print()  # Skip to the next line

            current_download_size = [0]
            GoogleDriveDownloader._save_response_content(response, dest_path, showsize,
                                                         current_download_size)
            print('Done.')

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, showsize, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if showsize:
                        print('\r' + sizeof_fmt(current_size[0]), end=' ')
                        sys.stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE


def download(url: str,
             path: Optional[str] = None,
             overwrite: Optional[bool] = False,
             sha1_hash: Optional[str] = None,
             retries: Optional[int] = 5,
             verify_ssl: Optional[bool] = True,
             anonymous_credential: Optional[bool] = True) -> str:
    """Download a given URL

    Parameters
    ----------
    url
        URL to download
    path
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite
        Whether to overwrite destination file if already exists.
    sha1_hash
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl
        Verify SSL certificates.
    anonymous_credential
        Whether to force to use anonymous credential if the path is from S3.

    Returns
    -------
    fname
        The file path of the downloaded file.
    """
    is_s3 = url.startswith(S3_PREFIX)
    if is_s3:
        boto3, botocore = try_import_boto3()
        s3 = boto3.resource('s3')
        if boto3.session.Session().get_credentials() is None or anonymous_credential:
            from botocore.handlers import disable_signing
            s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
        components = url[len(S3_PREFIX):].split('/')
        if len(components) < 2:
            raise ValueError('Invalid S3 url. Received url={}'.format(url))
        s3_bucket_name = components[0]
        s3_key = '/'.join(components[1:])
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0, currently it's {}".format(
        retries)

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not sha1sum(fname) == sha1_hash):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print('Downloading {} from {}...'.format(fname, url))
                if is_s3:
                    response = s3.meta.client.head_object(Bucket=s3_bucket_name,
                                                          Key=s3_key)
                    total_size = int(response.get('ContentLength', 0))
                    random_uuid = str(uuid.uuid4())
                    tmp_path = '{}.{}'.format(fname, random_uuid)
                    if tqdm is not None:
                        def hook(t_obj):
                            def inner(bytes_amount):
                                t_obj.update(bytes_amount)
                            return inner
                        with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True) as t:
                            s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path,
                                                         Callback=hook(t))
                    else:
                        s3.meta.client.download_file(s3_bucket_name, s3_key, tmp_path)
                else:
                    r = requests.get(url, stream=True, verify=verify_ssl)
                    if r.status_code != 200:
                        raise RuntimeError('Failed downloading url {}'.format(url))
                    # create uuid for temporary files
                    random_uuid = str(uuid.uuid4())
                    total_size = int(r.headers.get('content-length', 0))
                    chunk_size = 1024
                    if tqdm is not None:
                        t = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True)
                    with open('{}.{}'.format(fname, random_uuid), 'wb') as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:  # filter out keep-alive new chunks
                                if tqdm is not None:
                                    t.update(len(chunk))
                                f.write(chunk)
                    if tqdm is not None:
                        t.close()
                # if the target file exists(created by other processes)
                # and have the same hash with target file
                # delete the temporary file
                if not os.path.exists(fname) or (sha1_hash and not sha1sum(fname) == sha1_hash):
                    # atomic operation in the same file system
                    replace_file('{}.{}'.format(fname, random_uuid), fname)
                else:
                    try:
                        os.remove('{}.{}'.format(fname, random_uuid))
                    except OSError:
                        pass
                    finally:
                        warnings.warn(
                            'File {} exists in file system so the downloaded file is deleted'.format(fname))
                if sha1_hash and not sha1sum(fname) == sha1_hash:
                    raise UserWarning(
                        'File {} is downloaded but the content hash does not match.'
                        ' The repo may be outdated or download may be incomplete. '
                        'If the "repo_url" is overridden, consider switching to '
                        'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e

                print('download failed due to {}, retrying, {} attempt{} left'
                      .format(repr(e), retries, 's' if retries > 1 else ''))

    return fname


def check_version(min_version: str,
                  warning_only: bool = False,
                  library: Optional[ModuleType] = None):
    """Check the version of gluonnlp satisfies the provided minimum version.
    An exception is thrown if the check does not pass.

    Parameters
    ----------
    min_version
        Minimum version
    warning_only
        Printing a warning instead of throwing an exception.
    library
        The target library for version check. Checks gluonnlp by default
    """
    # pylint: disable=import-outside-toplevel
    from .. import __version__
    if library is None:
        version = __version__
        name = 'GluonNLP'
    else:
        version = library.__version__
        name = library.__name__
    from packaging.version import parse
    bad_version = parse(version.replace('.dev', '')) < parse(min_version)
    if bad_version:
        msg = 'Installed {} version {} does not satisfy the ' \
              'minimum required version {}'.format(name, version, min_version)
        if warning_only:
            warnings.warn(msg)
        else:
            raise AssertionError(msg)


def init_comm(backend, gpus):
    """Init communication backend

    Parameters
    ----------
    backend
        The communication backend
    gpus


    Returns
    -------
    store
        The kvstore
    num_workers
        The total number of workers
    rank
    local_rank
    is_master_node
    ctx_l
    """
    # backend specific implementation
    import mxnet as mx
    if backend == 'horovod':
        try:
            import horovod.mxnet as hvd  # pylint: disable=import-outside-toplevel
        except ImportError:
            logging.info('horovod must be installed.')
            sys.exit(1)
        hvd.init()
        store = None
        num_workers = hvd.size()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        is_master_node = rank == local_rank
        ctx_l = [mx.gpu(local_rank)]
        logging.info('GPU communication supported by horovod')
    else:
        store = mx.kv.create(backend)
        num_workers = store.num_workers
        rank = store.rank
        local_rank = 0
        is_master_node = rank == local_rank
        if gpus == '-1' or gpus == '':
            ctx_l = [mx.cpu()]
            logging.info('Runing on CPU')
        else:
            ctx_l = [mx.gpu(int(x)) for x in gpus.split(',')]
            logging.info('GPU communication supported by KVStore')

    return store, num_workers, rank, local_rank, is_master_node, ctx_l


def get_mxnet_visible_ctx():
    """Get the visible contexts in MXNet.

    - If GPU is available
        it will return all the visible GPUs, which can be controlled via "CUDA_VISIBLE_DEVICES".
    - If no GPU is available
        it will return the cpu device.

    Returns
    -------
    ctx_l
        The recommended contexts to use for MXNet
    """
    import mxnet as mx
    num_gpus = mx.context.num_gpus()
    if num_gpus == 0:
        ctx_l = [mx.cpu()]
    else:
        ctx_l = [mx.gpu(i) for i in range(num_gpus)]
    return ctx_l


# Python 3.9 feature backport https://github.com/python/cpython/pull/11478
class BooleanOptionalAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        if help is not None and default is not None:
            help += f" (default: {default})"

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)
