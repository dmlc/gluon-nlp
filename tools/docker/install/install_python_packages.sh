set -euo pipefail

pip3 install --no-cache --upgrade wheel

# python-dateutil==2.8.0 to satisfy botocore associated with latest awscli
pip3 install --no-cache --upgrade \
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
    awscli --user
