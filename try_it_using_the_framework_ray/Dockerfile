FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

WORKDIR /paper

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
    pip install deepspeed && \
    pip install evaluate && \
    pip install numpy && \
    pip install pillow

RUN pip install scikit-learn

RUN pip install ipython

RUN pip install --no-cache-dir -U "ray[all]"

RUN pip3 install cupy-cuda11x

RUN pip3 install jaxlib==0.3.22+cuda111.cudnn805 -f https://alpa-projects.github.io/wheels.html

#RUN pip install protobuf==3.20

ENTRYPOINT ["bash", "-c", "ray start --head --num-cpus 1 --dashboard-host 0.0.0.0 --include-dashboard true --block"]
