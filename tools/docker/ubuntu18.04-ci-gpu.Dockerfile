FROM gluonai/gluon-nlp:gpu-base-latest

LABEL maintainer="GluonNLP Team"

WORKDIR ${WORKDIR}/gluon-nlp
ADD gluon_nlp_job.sh .
RUN chmod +x gluon_nlp_job.sh
