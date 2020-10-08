FROM ubuntu:18.04

LABEL maintainer="GluonNLP Team"

ENV WORKDIR=/workspace
ENV SHELL=/bin/bash

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    ca-certificates \
    curl \
    emacs \
    subversion \
    locales \
    cmake \
    git \
    libopencv-dev \
    htop \
    vim \
    wget \
    unzip \
    libopenblas-dev \
    ninja-build \
    openssh-client \
    openssh-server \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libxft-dev \
    zlib1g-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s $(which python3) /usr/local/bin/python

# Install MXNet
RUN python3 -m pip install -U --pre "mxnet>=2.0.0b20200926" -f https://dist.mxnet.io/python --user

# Install PyTorch
RUN python3 -m pip install -U torch torchvision --user

RUN pip3 install --no-cache --upgrade \
    wheel \
    numpy==1.19.1 \
    pandas==0.25.1 \
    pytest \
    Pillow \
    requests==2.22.0 \
    scikit-learn==0.20.4 \
    scipy==1.2.2 \
    urllib3==1.25.8 \
    python-dateutil==2.8.0 \
    sagemaker-experiments==0.* \
    PyYAML==5.3.1 \
    mpi4py==3.0.2 \
    jupyterlab==2.2.4 \
    cmake \
    awscli

RUN mkdir -p ${WORKDIR}/notebook
RUN mkdir -p ${WORKDIR}/data
RUN cd ${WORKDIR} \
   && git clone https://github.com/dmlc/gluon-nlp \
   && cd gluon-nlp \
   && git checkout master \
   && python3 -m pip install -U -e ."[extras]" --user

COPY start_jupyter.sh /start_jupyter.sh
COPY devel_entrypoint.sh /devel_entrypoint.sh
RUN chmod +x /devel_entrypoint.sh

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

WORKDIR ${WORKDIR}

# Debug horovod by default
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf

# Install NodeJS + Tensorboard + TensorboardX
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt-get install -y nodejs

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libsndfile1-dev

RUN pip3 install --no-cache --upgrade \
    soundfile==0.10.2 \
    ipywidgets==7.5.1 \
    jupyter_tensorboard==0.2.0 \
    widgetsnbextension==3.5.1 \
    tensorboard==2.1.1 \
    tensorboardX==2.1
RUN jupyter labextension install jupyterlab_tensorboard \
   && jupyter nbextension enable --py widgetsnbextension \
   && jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Revise default shell to /bin/bash
RUN jupyter notebook --generate-config \
  && echo "c.NotebookApp.terminado_settings = { 'shell_command': ['/bin/bash'] }" >> /root/.jupyter/jupyter_notebook_config.py

# Add Tini
ARG TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT [ "/tini", "--", "/devel_entrypoint.sh" ]
CMD ["/bin/bash"]
