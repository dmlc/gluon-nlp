import pytest
import tempfile
import pandas as pd
from gluonnlp.cli.data.general_nlp_benchmark import prepare_glue


@pytest.mark.remote_required
@pytest.mark.parametrize('task', ["cola", "sst", "mrpc", "qqp", "sts", "mnli",
                                  "snli", "qnli", "rte", "wnli", "diagnostic"])
def test_glue(task):
    parser = prepare_glue.get_parser()
    with tempfile.TemporaryDirectory() as root:
        args = parser.parse_args(['--benchmark', 'glue',
                                  '--tasks', task,
                                  '--data_dir', root])
        prepare_glue.main(args)


@pytest.mark.remote_required
@pytest.mark.parametrize('task', ["cb", "copa", "multirc", "rte", "wic", "wsc", "boolq", "record",
                                  'broadcoverage-diagnostic', 'winogender-diagnostic'])
def test_glue(task):
    parser = prepare_glue.get_parser()
    with tempfile.TemporaryDirectory() as root:
        args = parser.parse_args(['--benchmark', 'superglue',
                                  '--tasks', task,
                                  '--data_dir', root])
        prepare_glue.main(args)
