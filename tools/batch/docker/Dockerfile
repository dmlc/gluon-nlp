FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      locales \
      cmake \
      wget \
      subversion \
      git \
      curl \
      vim \
      unzip \
      sudo \
      ca-certificates \
      libjpeg-dev \
      libpng-dev \
      libfreetype6-dev \
      python3-dev \
      python3-pip \
      python3-setuptools \
      libxft-dev &&\
  rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && pip3 install awscli && pip3 install --pre 'mxnet-cu102' -f https://dist.mxnet.io/python
RUN git clone https://github.com/dmlc/gluon-nlp
WORKDIR gluon-nlp
ADD gluon_nlp_job.sh .
