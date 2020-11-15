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

# ADD ask.py /QA
# ADD answer.py /QA
# ADD testingDocker /QA
ADD parse.py /QA
ADD answer /QA
ADD ask /QA
ADD distUtils.py /QA
ADD question.py /QA
## new ##
# Pull the environment name out of the environment.yml
# RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH
# ENV PATH /opt/conda/envs/nlp_qa/lib/python3.8
# ENV PYTHONPATH "${PYTHONPATH}:/opt/conda/envs/nlp_qa/"
ENV PATH="/opt/conda/envs/nlp_qa/bin:$PATH"
## 

CMD ["chmod 777 /QA/*"]

WORKDIR /QA

# SHELL ["conda", "run", "-n", "nlp_qa", "/bin/bash", "-c"]

ENTRYPOINT ["/bin/bash", "-c"]
# ENTRYPOINT ["conda", "run", "-n", "nlp_qa", "python", "testingDocker"]

# Make RUN commands use the new environment:
