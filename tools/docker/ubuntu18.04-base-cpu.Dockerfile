FROM ubuntu:18.04

LABEL maintainer="GluonNLP Team"
COPY install /install

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV WORKDIR=/workspace
ENV SHELL=/bin/bash

RUN mkdir -p ${WORKDIR}

RUN bash /install/install_ubuntu18.04_core.sh

RUN bash /install/install_python_packages.sh

# Install MXNet
RUN python3 -m pip install -U --pre "mxnet>=2.0.0b20200926" -f https://dist.mxnet.io/python --user

# Install PyTorch
RUN python3 -m pip install -U torch torchvision --user

# Install Jupyter Lab
RUN bash /install/install_jupyter_lab.sh

RUN mkdir -p ${WORKDIR}/data
RUN mkdir -p /.init
RUN cd ${WORKDIR} \
   && git clone https://github.com/dmlc/gluon-nlp \
   && cd gluon-nlp \
   && git checkout master \
   && python3 -m pip install -U -e ."[extras]" --user
