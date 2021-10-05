FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG USER_NAME

RUN test -n "$USER_NAME"
ARG UID_NUMBER
RUN test -n "$UID_NUMBER"
ARG GID_NUMBER
RUN test -n "$GID_NUMBER"

RUN apt-get update && apt-get -y install sudo

RUN groupadd $USER_NAME -g $GID_NUMBER
RUN adduser --shell /bin/bash --uid $UID_NUMBER --gid $GID_NUMBER \
    --disabled-password --gecos '' $USER_NAME
RUN adduser $USER_NAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

## }

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get -y install \
    arp-scan \
    build-essential \
    cmake \
    ffmpeg \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libsm6 \
    libswresample-dev \
    libswscale-dev \
    libxext6 \
    libxrender-dev \
    net-tools \
    pkg-config \
    protobuf-compiler \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    software-properties-common \
    tmux \
    wget


## Install Python 3.7 {
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get -y install \
        python3.7 \
        python3.7-dev \
        python3.7-tk
RUN sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 && \
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
## }

## Make SSH service available. Such service is required to use Pycharm 
## with remote interpreter. 
## Based on https://docs.docker.com/engine/examples/running_ssh_service/ {
ARG SSH_PASSWORD=my-secret-screencast
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo "$USER_NAME:$SSH_PASSWORD" | chpasswd
## The line below is commented out since it's not needed when you log in 
## as a non-root user and we created such user before in the Dockerfile.
# RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
## SSH login fix. Otherwise user is kicked out after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
## }

WORKDIR /home/$USER_NAME
## We're changing the user from the root to the given normal user.
## Note that if you want to run a command that only root can run, you have
## to prefix it with `sudo` from now on in the Dockerfile (an alternative is to
## change the user with `USER root`).
USER $USER_NAME

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
## Not sure why, but the `~/.local/bin` directory where the executable scripts
## installed by Python packages is not added automatically to PATH. Doint it
## manually here.
ENV PATH="/home/$USER_NAME/.local/bin:${PATH}"
ENV SHELL=/bin/bash

## Configure Jupyter notebook to produce Python code each time a notebook is
## saved
## TODO: when you create a new notebook in Jupyter lab, a corresponding empty
## `*.txt` file is created which is annoying. Not sure why this happens.
# COPY docker_auxiliary docker_auxiliary
# RUN jupyter notebook --generate-config && \
#     cat docker_auxiliary/jupyter_config_part.txt \
#         ~/.jupyter/jupyter_notebook_config.py > config_concatenated.py && \
#     mv config_concatenated.py ~/.jupyter/jupyter_notebook_config.py

## Additional information about how the image was built
ARG BUILD_INFO
RUN echo $BUILD_INFO > BUILD_INFO.txt
