FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

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

# Install Open MPI
RUN bash /install/install_openmpi.sh
ENV LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/openmpi/bin/:/usr/local/bin:/root/.local/bin:$PATH

RUN bash /install/install_python_packages.sh

# Install MXNet
RUN python3 -m pip install -U --pre "mxnet-cu102>=2.0.0b20200926" -f https://dist.mxnet.io/python --user

# Install PyTorch
RUN python3 -m pip install -U torch torchvision --user

# Install Horovod
RUN bash /install/install_horovod.sh

RUN mkdir -p ${WORKDIR}/notebook
RUN mkdir -p ${WORKDIR}/data
RUN mkdir -p /.init
RUN cd ${WORKDIR} \
   && git clone https://github.com/dmlc/gluon-nlp \
   && cd gluon-nlp \
   && git checkout master \
   && python3 -m pip install -U -e ."[extras]" --user

WORKDIR ${WORKDIR}
s