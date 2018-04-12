# Gluon NLP Toolkit

## Installation
1. Install nightly build of mxnet using `pip install --pre $MXNET_VARIANT` where `MXNET_VARIANT` can
   be any of the listed ones at https://github.com/apache/incubator-mxnet/issues/8671.
   Alternatively, you can also use the requirements files in this repo `pip install -r requirements.txt`
   for CPU-only MXNet, and `pip install -r requirements.gpu-cu90.txt` for GPU-enabled MXNet for CUDA
   9.0.
2. Clone this repo, and use `python setup.py install` for installing this package.
