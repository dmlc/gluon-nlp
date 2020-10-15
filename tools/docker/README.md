# Docker Support in GluonNLP
We provide the [Docker](https://www.docker.com/) container with everything set up to run GluonNLP.
With the prebuilt docker image, there is no need to worry about the operating systems or system dependencies. 
You can launch a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) development environment 
and try out to use GluonNLP to solve your problem.

| Name | Description | Target User |
|------|-------------|-------------|
| `cpu-ci-latest` or `gpu-ci-latest`   | Extends the CUDA image to include the basic functionalities, e.g., GluonNLP package, MXNet, PyTorch, Horovod. This is the image used in GluonNLP CI | GluonNLP Developers |  
| `cpu-latest` or `gpu-latest` | It has more functionality than the CI image, including the a development platform powered by Jupyter Lab. Some useful functionalities like Tensorboard are pre-installed. | Users that are willing to solve NLP problems and also do distributed training with Horovod + GluonNLP. |


## Run Docker
You can run the docker with the following command.

```
# On GPU machine
docker pull gluonai/gluon-nlp:gpu-latest
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --shm-size=2g gluonai/gluon-nlp:gpu-latest

# On CPU machine
docker pull gluonai/gluon-nlp:cpu-latest
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --shm-size=2g gluonai/gluon-nlp:cpu-latest
```

Here, we open the ports 8888, 8787, 8786, which are used for connecting to JupyterLab. 
Also, we set `--shm-size` to `2g`. This sets the shared memory storage to 2GB. Since NCCL will 
create shared memory segments, this argument is essential for the JupyterNotebook to work with NCCL. 
(See also https://github.com/NVIDIA/nccl/issues/290).

The folder structure of the docker image will be
```
/workspace/
├── gluonnlp
├── tvm
├── data
```

If you have a multi-GPU instance, e.g., [g4dn.12xlarge](https://aws.amazon.com/ec2/instance-types/g4/),
[p2.8xlarge](https://aws.amazon.com/ec2/instance-types/p2/),
[p3.8xlarge](https://aws.amazon.com/ec2/instance-types/p3/), you can try to verify the installation 
of horovod + MXNet by running the question answering script

```
# Assume that you are currently in GluonNLP

cd gluon-nlp/scripts/question_answering

docker run --gpus all --rm -it --shm-size=2g -v `pwd`:/workspace/data gluonai/gluon-nlp:gpu-latest \
    bash -c 'cd /workspace/data && bash commands/run_squad2_albert_base.sh 1 2.0'
```


## Build by yourself
To build a docker image from the dockerfile, you may use the following command:

```
# Build CPU Dockers
docker build -f ubuntu18.04-cpu.Dockerfile --target ci -t gluonai/gluon-nlp:cpu-ci-latest .
docker build -f ubuntu18.04-cpu.Dockerfile --target devel -t gluonai/gluon-nlp:cpu-latest .

# Build GPU Dockers
docker build -f ubuntu18.04-gpu.Dockerfile --target ci -t gluonai/gluon-nlp:gpu-ci-latest .
docker build -f ubuntu18.04-gpu.Dockerfile --target devel -t gluonai/gluon-nlp:gpu-latest .
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
path of MXNet by querying the MXNet runtime.

### Developers of GluonNLP
You may try to login to your dockerhub account and push the image to dockerhub.
```
docker push gluonai/gluon-nlp:cpu-ci-latest
docker push gluonai/gluon-nlp:cpu-latest

docker push gluonai/gluon-nlp:gpu-ci-latest
docker push gluonai/gluon-nlp:gpu-latest
```

### CI maintainer

Our current batch job dockers are in 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1. To
update the docker:
- Update the Dockerfile as described above
- Make sure docker and docker-compose, as well as the docker python package are installed.
- Export the AWS account credentials as environment variables
- CD to the same folder as the Dockerfile and execute the following:

```
# this executes a command that logs into ECR.
$(aws ecr get-login --no-include-email --region us-east-1)

# tags the recent build as gluon-nlp-1:latest, which AWS batch pulls from.
docker tag gluonai/gluon-nlp:gpu-ci-latest 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:gpu-ci-latest
docker tag gluonai/gluon-nlp:cpu-ci-latest 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:cpu-ci-latest

# pushes the change
docker push 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:gpu-ci-latest
docker push 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:cpu-ci-latest
```
