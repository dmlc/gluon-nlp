FROM ubuntu:18.04

LABEL maintainer="GluonNLP Team"
COPY install /install

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

# Install LLVM
RUN bash /install/install_llvm.sh

# Install Python Packages
RUN bash /install/install_python_packages.sh

# Install TVM
RUN bash /install/install_tvm_cpu.sh

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
   && python3 -m pip install -U -e ."[extras]"
