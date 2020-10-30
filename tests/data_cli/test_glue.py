import pytest
import tempfile
import pandas as pd
from gluonnlp.cli.data.general_nlp_benchmark import prepare_glue


@pytest.mark.remote_required
@pytest.mark.parametrize('benchmark', ['glue', 'superglue'])
def test_glue_superglue(benchmark):
    parser = prepare_glue.get_parser()
    with tempfile.TemporaryDirectory() as root:
        args = parser.parse_args(['--benchmark', benchmark,
                                  '--data_dir', root])
        prepare_glue.main(args)
