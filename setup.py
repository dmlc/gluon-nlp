#!/usr/bin/env python
import functools
import io
import os
import re

import numpy as np
import Cython.Build
from setuptools import Extension, find_packages, setup

Extension = functools.partial(
    Extension, include_dirs=[np.get_include()], define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")], language='c++',
    extra_compile_args=["-std=c++14"])

extensions = [
    Extension("gluonnlp.data.batchify.embedding",
              ["src/gluonnlp/data/batchify/embedding.pyx"]),
    Extension("gluonnlp.vocab.subwords", ["src/gluonnlp/vocab/subwords.pyx"])]
ext_modules = Cython.Build.cythonize(
    extensions, language_level='3str', compiler_directives={
        'binding': True, 'linetrace': True})


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")) as fp:
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

requirements = ['numpy']

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
    ext_modules=ext_modules,
    packages=find_packages(where="src", exclude=(
        'tests',
        'scripts',
    )),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements + ['cython>=0.29'],
    extras_require={
        'extras': [
            'spacy',
            'nltk==3.2.5',
            'sacremoses',
            'scipy',
            'numba>=0.40.1',
            'jieba',
            'sentencepiece',
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
