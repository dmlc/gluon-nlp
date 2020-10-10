set -e
set -u
set -o pipefail

apt-get update \
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

python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

ln -s $(which python3) /usr/local/bin/python
