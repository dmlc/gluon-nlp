#!/usr/bin/env python
from datetime import datetime
import io
import os
import re
import shutil
import sys
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('src', 'gluonnlp', '__init__.py')

if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')


requirements = [
    'numpy',
    'sacremoses>=0.0.38',
    'yacs>=0.1.6',
    'sacrebleu',
    'flake8',
    'packaging',
    'regex',
    'contextvars',
    'pyarrow',
    'sentencepiece==0.1.91',
    'protobuf',
    'pandas',
    'tokenizers==0.8.1',
    'click>=7.0',  # Dependency of youtokentome
    'youtokentome>=1.0.6',
    'fasttext>=0.9.1,!=0.9.2'  # Fix to 0.9.1 due to https://github.com/facebookresearch/fastText/issues/1052
]

setup(
    # Metadata
    name='gluonnlp',
    version=VERSION,
    python_requires='>=3.6',
    author='GluonNLP Toolkit Contributors',
    author_email='gluonnlp-dev@amazon.com',
    description='MXNet GluonNLP Toolkit (DeepNumpy Version)',
    long_description_content_type='text/markdown',
    license='Apache-2.0',

    # Package info
    packages=find_packages(where="src", exclude=(
        'tests',
        'scripts',
    )),
    package_dir={"": "src"},
    package_data={'': [os.path.join('models', 'model_zoo_checksums', '*.txt'),
                       os.path.join('cli', 'data', 'url_checksums', '*.txt'),
                       os.path.join('cli', 'data', 'url_checksums', 'mirror', '*.json')]},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'extras': [
            'boto3',
            'tqdm',
            'jieba',
            'subword_nmt',
            'spacy>=2.3.0',
            'langid==1.1.6',
            'nltk',
            'h5py>=2.10',
            'scipy',
            'tqdm'
        ],
        'dev': [
            'pytest',
            'pytest-env',
            'pylint',
            'pylint_quotes',
            'flake8',
            'recommonmark',
            'sphinx-gallery',
            'sphinx_rtd_theme',
            'mxtheme',
            'sphinx-autodoc-typehints',
            'nbsphinx',
            'flaky',
        ],
    },

    entry_points={
        'console_scripts': [
            'nlp_data = gluonnlp.cli.data.__main__:cli_main',
            'nlp_process = gluonnlp.cli.process.__main__:cli_main',
            'gluon_average_checkpoint = gluonnlp.cli.average_checkpoint:cli_main'
        ],
    },
)
