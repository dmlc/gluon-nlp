#!/usr/bin/env python
import io
import os
import re
import shutil
import sys
from setuptools import setup, find_packages, Extension


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


readme = io.open('README.rst', encoding='utf-8').read()

VERSION = find_version('src', 'gluonnlp', '__init__.py')

requirements = [
    'numpy>=1.16.0',
    'cython',
    'packaging'
]

setup(
    # Metadata
    name='gluonnlp',
    version=VERSION,
    python_requires='>=3.5',
    author='Gluon NLP Toolkit Contributors',
    author_email='mxnet-gluon@amazon.com',
    url='https://github.com/dmlc/gluon-nlp',
    description='MXNet Gluon NLP Toolkit',
    long_description=readme,
    long_description_content_type='text/x-rst',
    license='Apache-2.0',

    # Package info
    packages=find_packages(where="src", exclude=(
        'tests',
        'scripts',
    )),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
    ],
    install_requires=requirements,
    extras_require={
        'extras': [
            'spacy',
            'nltk',
            'sacremoses',
            'scipy',
            'numba>=0.45',
            'jieba',
            'sentencepiece',
            'boto3',
            'tqdm',
            'sacremoses',
            'regex',
            'packaging',
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
    ext_modules=[
        Extension('gluonnlp.data.fast_bert_tokenizer', sources=['src/gluonnlp/data/fast_bert_tokenizer.pyx']),
    ],
)
