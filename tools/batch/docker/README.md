# Updating the Docker for AWS Batch.

Our current batch job dockers are in 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1. To
update the docker:
- update the Dockerfile
- Make sure docker and docker-compose, as well as the docker python package are installed.
- Export the AWS account credentials as environment variables
- CD to the same folder as the Dockerfile and execute the following:

```
# this executes a command that logs into ECR.
$(aws ecr get-login --no-include-email --region us-east-1)

# builds the Dockerfile as gluon-nlp-1 docker.
docker build -f Dockerfile.gpu -t gluon-nlp-1:gpu .
docker build -f Dockerfile.cpu -t gluon-nlp-1:cpu .

# tags the recent build as gluon-nlp-1:latest, which AWS batch pulls from.
docker tag gluon-nlp-1:gpu 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:latest
docker tag gluon-nlp-1:cpu 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:cpu-latest

# pushes the change
docker push 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:latest
docker push 747303060528.dkr.ecr.us-east-1.amazonaws.com/gluon-nlp-1:cpu-latest
```
