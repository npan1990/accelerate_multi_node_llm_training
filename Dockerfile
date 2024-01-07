FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

WORKDIR /app

ENV TZ="Europe/Athens"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y mpich && \
    pip install torch torchvision torchaudio && \
    pip install transformers && \
    pip install accelerate && \
    pip install datasets && \
    pip install loguru && \
    pip install click && \
    pip install tqdm && \
    pip install deepspeed

COPY . .

RUN apt-get update && \
    apt-get install -y openssh-server

# RUN mkdir /root/.ssh/

COPY ssh/id_rsa /root/.ssh/id_rsa

COPY ssh/id_rsa.pub /root/.ssh/id_rsa.pub

COPY ssh/id_rsa.pub /root/.ssh/authorized_keys

RUN chmod 600 /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa

EXPOSE 22

RUN apt-get update && \
    apt-get install -y ufw && \
    ufw allow 22

RUN service ssh start

ENTRYPOINT service ssh start && sleep infinity