"""
setup.py: prepares required libraries for BERT scripts
"""
#!/usr/bin/env python
import pathlib
import sys
import os
import logging
from setuptools import setup
import mxnet

requirements = [
    'numpy>=1.16.0',
]

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
out_lib_file = 'bertpass_lib.so'
log.info(' ... compiling BERT custom graph pass library into %s', out_lib_file)
mxnet_path = pathlib.Path(mxnet.__file__).parent.absolute()
mxnet_include_path = pathlib.Path.joinpath(mxnet_path, 'include/mxnet')
os.system('g++ -shared -fPIC -std=c++11 bertpass_gpu.cc -o bertpass_lib.so -I ' +
          str(mxnet_include_path))
