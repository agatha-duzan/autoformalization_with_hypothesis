FROM --platform=linux/amd64 nvidia/cuda:12.1.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-cu"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl git wget gnupg python3 python3-pip python-is-python3

# install lean
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
ENV PATH=$PATH:$HOME/.elan/bin
# RUN echo "source \$HOME/.elan/env" >> ~/.bashrc
# RUN echo "source \$HOME/.elan/env" >> ~/.profile

# mount current workspace to /workspace
WORKDIR /workspace
COPY . /workspace

# prepare lean REPL
RUN cd /workspace/repl && $HOME/.elan/bin/lake exe cache get

# install python packages
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade build
RUN python3 -m build
RUN pip3 install -e .
