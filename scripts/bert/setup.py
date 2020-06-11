#!/usr/bin/env python
import mxnet
import pathlib
import sys
import os
import logging

from setuptools import setup

requirements = [
    'numpy>=1.16.0',
],

setup(
    # Metadata
    name='gluonnlp-scripts-bert',
    python_requires='>=3.5',
    author='Gluon NLP Toolkit Contributors',
    author_email='mxnet-gluon@amazon.com',
    url='https://github.com/dmlc/gluon-nlp',
    description='MXNet Gluon NLP Toolkit - BERT scripts',
    license='Apache-2.0'
)

# compile custom graph pass for bert
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger()
log.info(' ... compiling BERT custom graph pass library')
out_lib_file = 'bertpass_lib.so'
mxnet_path = pathlib.Path(mxnet.__file__).parent.absolute()
mxnet_include_path = pathlib.Path.joinpath(mxnet_path, 'include/mxnet')
os.system('g++ -shared -fPIC -std=c++11 bertpass_gpu.cc -o bertpass_lib.so -I ' + mxnet_include_path)
log.info(' Done: library was created in: %s', out_lib_file)

