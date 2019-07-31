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
 RUN /bin/bash -c 'CONDA_ENVS_PATH=$PWD/conda CONDA_PKGS_DIRS=$PWD/conda/pkgs conda init bash && source /root/.bashrc && conda env update --prune -p conda/gpu/py3 -f env/gpu/py3.yml && source activate ./conda/gpu/py3 && pip install -v -e . && pip install awscli && python -m spacy download en && python -m spacy download de && python -m nltk.downloader all'
 ADD gluon_nlp_job.sh .
