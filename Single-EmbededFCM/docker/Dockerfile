FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt -y update && apt-get -y install python3-pip python3-dev python3-setuptools
RUN pip3 install --upgrade pip
RUN apt-get install -y python3-dev python3-venv python3-tk

RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-cache policy python3.8
RUN apt install -y python3.8
RUN apt install -y python3.8-dev python3.8-venv python3.8-tk 

COPY set_env.sh /workspace/set_env.sh

WORKDIR /workspace
