#!/usr/bin/env python
from datetime import datetime
import io
import os
import re
import shutil
import sys
import warnings
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('src', 'gluonnlp', '__init__.py')

if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')

requirements = [
    'boto3',
    'numpy<1.20.0',
    'sacremoses>=0.0.38,<0.0.44',
    'yacs>=0.1.6',
    'sacrebleu',
    'flake8',
    'packaging',
    'regex',
    'contextvars',
    'pyarrow',
    'sentencepiece==0.1.95',
    'protobuf',
    'pandas',
    'tokenizers==0.9.4',
    'dataclasses;python_version<"3.7"',  # Dataclass for python <= 3.6
    'pickle5;python_version<"3.8"',  # pickle protocol 5 for python <= 3.8
    'click>=7.0',  # Dependency of youtokentome
    'youtokentome>=1.0.6',
    'fasttext>=0.9.1,!=0.9.2'  # Fix to 0.9.1 due to https://github.com/facebookresearch/fastText/issues/1052
]

extensions = []
cmdclass = {}
try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    force_cuda = os.getenv("FORCE_TORCH_CUDA", "0") == "1"
    if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
        extensions.extend([
            CUDAExtension(
                name="gluonnlp.torch.fused_optimizers",
                include_dirs=[
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "src/gluonnlp/torch/clib")
                ],
                sources=[
                    "src/gluonnlp/torch/clib/amp_C_frontend.cpp",
                    "src/gluonnlp/torch/clib/multi_tensor_lans.cu",
                    "src/gluonnlp/torch/clib/multi_tensor_l2norm_kernel.cu"
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math"]
                },
            )
        ])
        cmdclass["build_ext"] = BuildExtension
    else:
        warnings.warn("Cannot install fused cuda optimizers.")
except ImportError:
    pass

setup(
    # Metadata
    name='gluonnlp',
    version=VERSION,
    python_requires='>=3.6',
    author='GluonNLP Toolkit Contributors',
    author_email='gluonnlp-dev@amazon.com',
    description='GluonNLP Toolkit',
    long_description_content_type='text/markdown',
    license='Apache-2.0',

    # Package info
    packages=find_packages(where="src", exclude=(
        'tests',
        'scripts',
    )),
    package_dir={"": "src"},
    package_data={
        '': [
            os.path.join('models', 'model_zoo_checksums', '*.txt'),
            os.path.join('cli', 'data', 'url_checksums', '*.txt'),
            os.path.join('cli', 'data', 'url_checksums', 'mirror', '*.json')
        ]
    },
    zip_safe=True,
    include_package_data=True,
    ext_modules=extensions,
    cmdclass=cmdclass,
    install_requires=requirements,
    extras_require={
        'extras': [
            'tqdm',
            'jieba',
            'subword_nmt',
            'spacy>=2.3.0,<3',
            'langid==1.1.6',
            'nltk',
            'h5py>=2.10',
            'scipy',
            'wikiextractor>=3.0.4,<4',
            'tqdm',
            'py3nvml',
            'smart_open',
        ],
        'dev': [
            'pytest',
            'pytest-env',
            'pytest-mock',
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
        'web': [
            'ipython',
            'sphinx>=1.5.5',
            'sphinx-gallery',
            'nbsphinx',
            'sphinx_rtd_theme',
            'mxtheme',
            'sphinx-autodoc-typehints',
            'matplotlib',
            'Image',
            'recommonmark',
            'nbformat',
            'notedown',
            'jupyter_client',
            'ipykernel',
            'matplotlib',
            'termcolor',
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
