import pytest
import tempfile
import os
import mxnet as mx
import multiprocessing
import functools
from mxnet.gluon import nn
import numpy as np
from numpy.testing import assert_allclose
from gluonnlp.utils.misc import AverageSGDTracker, download, sha1sum
mx.npx.set_np()


def test_average_sgd_tracker():
    samples = [mx.np.random.normal(0, 1, (10, 3)) for _ in range(10)]
    no_moving_avg_param_l = []
    with_moving_avg_param_l = []
    moving_avg_param = None
    net_final_moving_avg_param = None
    for use_moving_avg in [False, True]:
        net = nn.HybridSequential(prefix='net_')
        with net.name_scope():
            net.add(nn.Dense(10))
            net.add(nn.Dense(3))
        net.initialize(init=mx.init.One())
        net.hybridize()
        trainer = mx.gluon.Trainer(net.collect_params(), 'adam')
        if use_moving_avg:
            model_averager = AverageSGDTracker(net.collect_params())
        for sample in samples:
            out = sample ** 3 + sample
            with mx.autograd.record():
                pred = net(sample)
                loss = ((out - pred) ** 2).mean()
            loss.backward()
            trainer.step(1.0)
            if use_moving_avg:
                model_averager.step()
                print(model_averager.average_params)
            if use_moving_avg:
                with_moving_avg_param_l.append({k: v.data().asnumpy() for k, v in net.collect_params().items()})
            else:
                no_moving_avg_param_l.append({k: v.data().asnumpy() for k, v in net.collect_params().items()})
        if use_moving_avg:
            model_averager.copy_back()
            moving_avg_param = {k: v.asnumpy() for k, v in model_averager.average_params.items()}
            net_final_moving_avg_param = {k: v.data().asnumpy() for k, v in net.collect_params().items()}
    # Match the recorded params
    calculated_moving_param = {k: np.zeros(v.shape) for k, v in no_moving_avg_param_l[0].items()}
    for step, (no_moving_avg_param, with_moving_avg_param) in enumerate(zip(no_moving_avg_param_l,
                                                                            with_moving_avg_param_l)):
        decay = 1.0 / (step + 1)
        assert len(no_moving_avg_param) == len(with_moving_avg_param)
        for k in with_moving_avg_param:
            assert_allclose(no_moving_avg_param[k], with_moving_avg_param[k])
            calculated_moving_param[k] += decay * (with_moving_avg_param[k] - calculated_moving_param[k])
    assert len(moving_avg_param) == len(net_final_moving_avg_param)
    for k in moving_avg_param:
        assert_allclose(moving_avg_param[k], calculated_moving_param[k], 1E-5, 1E-5)
        assert_allclose(moving_avg_param[k], net_final_moving_avg_param[k], 1E-5, 1E-5)


def s3_enabled():
    from gluonnlp.utils.lazy_imports import try_import_boto3
    try:
        boto3 = try_import_boto3()
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
