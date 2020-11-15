# Ubuntu Linux as the base image
FROM ubuntu:18.04

# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

FROM continuumio/miniconda3

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

RUN mkdir /QA

# Adding in the actual python files, answer and ask are execs
ADD parse.py /QA
ADD answer /QA
ADD ask /QA
ADD distUtils.py /QA
ADD question.py /QA

# Change the python path to look at the nlp_qa env 
ENV PATH="/opt/conda/envs/nlp_qa/bin:$PATH"

# Changing the permissions to be executable for all
CMD ["chmod 777 /QA/*"]

WORKDIR /QA


ENTRYPOINT ["/bin/bash", "-c"]

