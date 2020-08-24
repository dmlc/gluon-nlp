# Docker Support in GluonNLP
We provide the [Docker](https://www.docker.com/) container with everything set up to run GluonNLP.
With the prebuilt docker image, there is no need to worry about the operating systems or system dependencies. 
You can launch a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) development environment 
and try out to use GluonNLP to solve your problem.

## Run Docker
You can run the docker with the following command.

```
docker pull gluonai/gluon-nlp:gpu-latest
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --shm-size=2g gluonai/gluon-nlp:gpu-latest
```

Here, we open the ports 8888, 8787, 8786, which are used for connecting to JupyterLab. 
Also, we set `--shm-size` to `2g`. This sets the shared memory storage to 2GB. Since NCCL will 
create shared memory segments, this argument is essential for the JupyterNotebook to work with NCCL. 
(See also https://github.com/NVIDIA/nccl/issues/290).

The folder structure of the docker image will be
```
/workspace/
├── gluonnlp
├── horovod
├── mxnet
├── notebooks
├── data
```

If you have a multi-GPU instance, e.g., [g4dn.12xlarge](https://aws.amazon.com/ec2/instance-types/g4/),
[p2.8xlarge](https://aws.amazon.com/ec2/instance-types/p2/),
[p3.8xlarge](https://aws.amazon.com/ec2/instance-types/p3/), you can try to run the following 
command to verify the installation of horovod + MXNet

```
docker run --gpus all --rm -it --shm-size=4g gluonai/gluon-nlp:gpu-latest \
    horovodrun -np 2 python3 -m pytest /workspace/horovod/horovod/test/test_mxnet.py
```


## Build your own Docker Image
To build a docker image fom the dockerfile, you may use the following command:

```
docker build -f ubuntu18.04-devel-gpu.Dockerfile -t gluonai/gluon-nlp:gpu-latest .
```
