FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as base

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
ENV PATH=/usr/local/openmpi/bin/:/usr/local/bin:/root/.local/bin:/usr/bin:$PATH

# Install LLVM
RUN bash /install/install_llvm.sh

# Install Python Packages
RUN bash /install/install_python_packages.sh

# Install TVM
RUN bash /install/install_tvm_gpu.sh

# Install MXNet
RUN python3 -m pip install -U --pre "mxnet-cu102>=2.0.0a" --user

# Install PyTorch
RUN python3 -m pip install "torch==1.8.1+cu102" torchvision -f https://download.pytorch.org/whl/torch_stable.html

# Install Horovod
RUN bash /install/install_horovod.sh

# Install Jupyter Lab
RUN bash /install/install_jupyter_lab.sh

RUN mkdir -p ${WORKDIR}/data
RUN mkdir -p /.init
RUN cd ${WORKDIR} \
   && git clone https://github.com/dmlc/gluon-nlp \
   && cd gluon-nlp \
   && git checkout master \
   && python3 -m pip install -U -e ."[extras,dev]"

# Stage-CI
FROM base as ci
WORKDIR ${WORKDIR}/gluon-nlp
ADD gluon_nlp_job.sh .
RUN chmod +x gluon_nlp_job.sh

# Stage-Devel
FROM base as devel
COPY start_jupyter.sh /start_jupyter.sh
COPY devel_entrypoint.sh /devel_entrypoint.sh
RUN chmod +x /devel_entrypoint.sh

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

WORKDIR ${WORKDIR}

# Add Tini
ARG TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT [ "/tini", "--", "/devel_entrypoint.sh" ]
CMD ["/bin/bash"]
