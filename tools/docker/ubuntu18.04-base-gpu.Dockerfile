FROM gluonai/gluon-nlp:cspu-base-latest

LABEL maintainer="GluonNLP Team"

COPY start_jupyter.sh /start_jupyter.sh
COPY devel_entrypoint.sh /devel_entrypoint.sh
RUN chmod +x /devel_entrypoint.sh

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

WORKDIR ${WORKDIR}

# Add Tini
ARG TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT [ "/tini", "--", "/devel_entrypoint.sh" ]
CMD ["/bin/bash"]
