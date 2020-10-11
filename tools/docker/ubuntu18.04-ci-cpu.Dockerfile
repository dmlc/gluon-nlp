FROM gluonai/gluon-nlp:cpu-base-latest

LABEL maintainer="GluonNLP Team"

WORKDIR ${WORKDIR}/gluon-nlp
ADD gluon_nlp_cpu_job.sh .
RUN chmod +x gluon_nlp_cpu_job.sh
