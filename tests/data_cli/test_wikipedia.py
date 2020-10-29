import pytest
import tempfile
from gluonnlp.cli.data.pretrain_corpus import prepare_wikipedia


@pytest.mark.remote_required
# Test for zh-classical (文言) + wuu (吴语), which are smaller compared with English
@pytest.mark.parametrize('lang', ['zh-classical', 'wuu'])
def test_download_format(lang):
    parser = prepare_wikipedia.get_parser()
    with tempfile.TemporaryDirectory() as root:
        download_args = parser.parse_args(['--mode', 'download+format',
                                           '--lang', lang,
                                           '--date', 'latest', '-o', root])
        prepare_wikipedia.main(download_args)
