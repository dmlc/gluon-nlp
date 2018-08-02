#!/usr/bin/env python
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


readme = io.open('README.rst', encoding='utf-8').read()

VERSION = find_version('gluonnlp', '__init__.py')

requirements = [
    'numpy',
]

setup(
    # Metadata
    name='gluonnlp',
    version=VERSION,
    author='Gluon NLP Toolkit Contributors',
    author_email='mxnet-gluon@amazon.com',
    url='https://github.com/dmlc/gluon-nlp',
    description='MXNet Gluon NLP Toolkit',
    long_description=readme,
    license='Apache-2.0',

    # Package info
    packages=find_packages(exclude=(
        'tests',
        'scripts',
    )),
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'extras': [
            'spacy',
            'nltk==3.2.5',
            'sacremoses',
            'scipy',
            'numba',
            'jieba',
        ],
        'dev': [
            'pytest',
            'recommonmark',
            'sphinx-gallery',
            'sphinx_rtd_theme',
            'nbsphinx',
        ],
    },
)
