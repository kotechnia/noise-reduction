ARG CUDA_DOCKER_VERSION=11.7.1-cudnn8-runtime-ubuntu20.04

FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install curl -y
RUN apt-get install git -y
RUN apt-get install python3-pip -y

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get install libsndfile1-dev -y --no-install-recommends

RUN mkdir -p /root/workspace
WORKDIR /root/workspace
