# Docker Support in GluonNLP
We provide the [Docker](https://www.docker.com/) container with everything set up to run GluonNLP.
With the prebuilt docker image, there is no need to worry about the operating systems or system dependencies. 
You can launch a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) development environment 
and try out to use GluonNLP to solve your problem.

## Run Docker
You can run the docker with the following command:

```
docker pull gluonai/gluon-nlp:v1.0.0
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 gluonai/gluon-nlp:v1.0.0
```


## Build your own Docker Image
To build a docker image fom the dockerfile, you may use the following command:

```
docker build -f ubuntu18.04-devel-gpu.Dockerfile -t gluonai/gluon-nlp .
```
