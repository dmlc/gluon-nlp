# Docker Support in GluonNLP
We provide the [Docker](https://www.docker.com/) container with everything set up to run GluonNLP.
With the prebuilt docker image, there is no need to worry about the operating systems or system dependencies. 
You can launch a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) development environment 
and try out to use GluonNLP to solve your problem.

| Name | Description | Target User |
|------|-------------|-------------|
| `base` | Extends the CUDA image to include the basic functionalities, e.g., GluonNLP package, MXNet, PyTorch, Horovod. You can directly use the docker to run distributed training jobs. | Users that are willing to use GluonNLP to train models. For example, you can use the docker image for distributed training. |  
| `devel` | Extends the base image to include a development platform powered by Jupyter Lab. Some useful functionalities like Tensorboard are pre-installed. | Users that are willing to analyze NLP data and build models with GluonNLP. |


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
├── data
```

If you have a multi-GPU instance, e.g., [g4dn.12xlarge](https://aws.amazon.com/ec2/instance-types/g4/),
[p2.8xlarge](https://aws.amazon.com/ec2/instance-types/p2/),
[p3.8xlarge](https://aws.amazon.com/ec2/instance-types/p3/), you can try to verify the installation 
of horovod + MXNet by running the question answering script

```
docker run --gpus all --rm -it --shm-size=4g gluonai/gluon-nlp:gpu-latest \
    horovodrun -np 2 python3 -m pytest /workspace/horovod/horovod/test/test_mxnet.py
```


## Build your own Docker Image
To build a docker image fom the dockerfile, you may use the following command:

```
docker build -f ubuntu18.04-base-cpu.Dockerfile -t gluonai/gluon-nlp:cpu-base-latest .
docker build -f ubuntu18.04-devel-cpu.Dockerfile -t gluonai/gluon-nlp:cpu-latest .

docker build -f ubuntu18.04-base-gpu.Dockerfile -t gluonai/gluon-nlp:gpu-base-latest .
docker build -f ubuntu18.04-devel-gpu.Dockerfile -t gluonai/gluon-nlp:gpu-latest .
```

In addition, to build the GPU docker, you will need to install the nvidia-docker2 and edit `/etc/docker/daemon.json` like the following:

```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

After that, restart docker via `sudo systemctl restart docker.service`.

For more details, you may refer to https://github.com/NVIDIA/nvidia-docker/issues/595. We need this additional setup
because the horovod+mxnet integration identifies the library and include 
path of MXNet by querying th MXNet runtime.

### Developers of GluonNLP
You may try to login to your dockerhub account and push the image to dockerhub.
```
docker push gluonai/gluon-nlp:gpu-latest
```
