import os
import pytest
from gluonnlp.cli.data.pretrain_corpus import prepare_wikipedia


@pytest.mark.remote_required
# Test for zh-classical + hindi, which are smaller compared with English
@pytest.mark.parametrize('lang', ['zh-classical', 'hi'])
def test_download_format(lang):
    parser = prepare_wikipedia.get_parser()
    download_args = parser.parse_args(['--mode', 'download', '--lang', lang, '--date', 'latest',
                                       '-'])
    prepare_wikipedia.main(download_args)
