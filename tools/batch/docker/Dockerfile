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

RUN pip3 install --upgrade pip
RUN pip3 install  --no-cache --upgrade \
    wheel \
    cmake \
    awscli
RUN mkdir workspace
WORKDIR workspace
ADD gluon_nlp_job.sh .
