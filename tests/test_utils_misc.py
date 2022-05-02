import pytest
import tempfile
import os
import logging
import multiprocessing
import functools
from pathlib import Path
import numpy as np
import mxnet as mx
from gluonnlp.utils.misc import download, sha1sum, logging_config,\
    get_mxnet_visible_device, logerror



def s3_enabled():
    from gluonnlp.utils.lazy_imports import try_import_boto3
    try:
        boto3, botocore = try_import_boto3()
        s3 = boto3.resource('s3')
        for bucket in s3.buckets.all():
            print(bucket.name)
        return True
    except Exception:
        return False


def verify_download(url, sha1_hash, overwrite):
    with tempfile.TemporaryDirectory() as root:
        download_path = os.path.join(root, 'dat0')
        # Firstly, verify that we are able to get download the data correctly
        download(url, sha1_hash=sha1_hash, path=download_path, overwrite=overwrite)
        assert sha1sum(download_path) == sha1_hash
        os.remove(download_path)

        # Secondly, verify that we are able to download with multiprocessing
        download_path = os.path.join(root, 'dat1')
        with multiprocessing.Pool(2) as pool:
            pool.map(functools.partial(download, sha1_hash=sha1_hash,
                                       path=download_path, overwrite=overwrite),
                     [url for _ in range(2)])
        assert sha1sum(download_path) == sha1_hash
        os.remove(download_path)


@pytest.mark.skipif(not s3_enabled(),
                    reason='S3 is not supported. So this test is skipped.')
@pytest.mark.parametrize('overwrite', [False, True])
@pytest.mark.remote_required
def test_download_s3(overwrite):
    verify_download(url='s3://commoncrawl/crawl-data/CC-MAIN-2014-41/cc-index.paths.gz',
                    sha1_hash='fac65325fdd881b75d6badc0f3caea287d91ed54',
                    overwrite=overwrite)


@pytest.mark.remote_required
@pytest.mark.parametrize('overwrite', [False, True])
def test_download_https(overwrite):
    verify_download(url='https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2014-41/'
                        'cc-index.paths.gz',
                    sha1_hash='fac65325fdd881b75d6badc0f3caea287d91ed54',
                    overwrite=overwrite)


@pytest.mark.remote_required
@pytest.mark.parametrize('overwrite', [False, True])
def test_download_non_existing(overwrite):
    with pytest.raises(RuntimeError, match='Failed downloading url'):
        verify_download(url='https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2014-41/non_existing',
                        sha1_hash='foo',
                        overwrite=overwrite)


def test_logging_config():
    logger = logging.getLogger(__name__)
    with tempfile.TemporaryDirectory() as root:
        logging_config(folder=root, logger=logger, name='test')
        file_names = os.listdir(root)
        assert file_names[0] == 'test.log'
        file_size = Path(os.path.join(root, 'test.log')).stat().st_size
        assert file_size == 0
        logger.info('123')
        for handler in logger.handlers:
            handler.flush()
        file_size_test1 = Path(os.path.join(root, 'test.log')).stat().st_size
        assert file_size_test1 > 0
        logging_config(folder=root, logger=logger, name='foo', overwrite_handler=False)
        logger.info('123')
        for handler in logger.handlers:
            handler.flush()
        file_size_test2 = Path(os.path.join(root, 'test.log')).stat().st_size
        file_size_foo1 = Path(os.path.join(root, 'foo.log')).stat().st_size
        assert file_size_test2 > file_size_test1
        assert file_size_foo1 > 0

        # After overwrite, the old hanlder will be removed
        logging_config(folder=root, logger=logger, name='zoo', overwrite_handler=True)
        logger.info('12345')
        for handler in logger.handlers:
            handler.flush()
        file_size_zoo1 = Path(os.path.join(root, 'zoo.log')).stat().st_size
        file_size_test3 = Path(os.path.join(root, 'test.log')).stat().st_size
        file_size_foo2 = Path(os.path.join(root, 'foo.log')).stat().st_size
        assert file_size_test3 == file_size_test2
        assert file_size_foo2 == file_size_foo1
        assert file_size_zoo1 > 0


def test_get_mxnet_visible_device(device):
    device_l = get_mxnet_visible_device()
    for ele_device in device_l:
        arr = mx.np.array(1.0, device=ele_device)
        arr.asnumpy()


@pytest.mark.parametrize('a,err', [
    (None, TypeError),
    (range(2), IndexError)])
def test_logerror(mocker, a, err):
    logger = logging.getLogger(__name__)
    spy = mocker.spy(logger, 'exception')

    @logerror(logger=logger)
    def test_fn(a):
        return a[3]

    with pytest.raises(err):
        test_fn(a)

    assert spy.call_count == 1
