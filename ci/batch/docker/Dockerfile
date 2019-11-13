FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

 RUN apt-get update && apt-get install -y --no-install-recommends \
          build-essential \
          locales \
          cmake \
          git \
          curl \
          vim \
          unzip \
          sudo \
          ca-certificates \
          libjpeg-dev \
          libpng-dev \
          libfreetype6-dev \
          libxft-dev &&\
      rm -rf /var/lib/apt/lists/*

 RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
      chmod +x ~/miniconda.sh && \
      ~/miniconda.sh -b -p /opt/conda && \
      rm ~/miniconda.sh && \
      /opt/conda/bin/conda clean -ya
 ENV PATH /opt/conda/bin:$PATH
 RUN git clone https://github.com/dmlc/gluon-nlp
 WORKDIR gluon-nlp
 ADD gluon_nlp_job.sh .
