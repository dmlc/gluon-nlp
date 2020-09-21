"""
setup.py: prepares required libraries for BERT scripts
"""
#!/usr/bin/env python
import pathlib
import sys
import os
import logging
from distutils.command.install import install
from setuptools import setup
import mxnet

requirements = [
    'numpy>=1.16.0',
]

def CompileBERTCustomPass():
    """Compiles custom graph pass for BERT into a library. It offers performance improvements"""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    log = logging.getLogger()
    input_pass_file = 'bertpass_gpu.cc'
    out_lib_file = 'bertpass_lib.so'
    log.info(' ... compiling BERT custom graph pass into %s', out_lib_file)
    mxnet_path = pathlib.Path(mxnet.__file__).parent.absolute()
    mxnet_include_path = pathlib.Path.joinpath(mxnet_path, 'include')
    pass_path = os.path.dirname(os.path.realpath(__file__))
    source = os.path.join(pass_path, input_pass_file)
    target = os.path.join(pass_path, out_lib_file)
    os.system('g++ -shared -fPIC -std=c++11 ' + str(source) +
              ' -o ' + str(target) + ' -I ' +
              str(mxnet_include_path))

class CompileBERTPass(install):
    def run(self):
        install.run(self)
        self.execute(CompileBERTCustomPass, ())

setup(
    # Metadata
    name='gluonnlp-scripts-bert',
    python_requires='>=3.5',
    author='Gluon NLP Toolkit Contributors',
    author_email='mxnet-gluon@amazon.com',
    url='https://github.com/dmlc/gluon-nlp',
    description='MXNet Gluon NLP Toolkit - BERT scripts',
    license='Apache-2.0',
    cmdclass={'install': CompileBERTPass}
)
