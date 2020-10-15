set -euo pipefail

# Install Horovod
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITHOUT_GLOO=1 \
HOROVOD_WITH_MPI=1 HOROVOD_WITH_MXNET=1 HOROVOD_WITH_PYTORCH=1 \
HOROVOD_WITHOUT_TENSORFLOW=1 python3 -m pip install --no-cache-dir horovod==0.20.3 --user
# Debug horovod by default
echo NCCL_DEBUG=INFO >> /etc/nccl.conf
