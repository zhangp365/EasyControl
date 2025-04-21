FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS env_base
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN --mount=type=cache,target=/var/cache/apt \
apt-get update && apt-get install -y git vim libgl1-mesa-glx libglib2.0-0 python3-dev gcc g++ && apt-get clean
RUN pip3 install --no-cache-dir --upgrade pip setuptools

FROM env_base AS base 
RUN git clone https://github.com/zhangp365/EasyControl.git /EasyControl

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
# Install comfyUI
RUN sed -i '/torch/d' /EasyControl/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip3 install --no-cache-dir git+https://github.com/huggingface/optimum-quanto.git
RUN --mount=type=cache,target=/root/.cache/pip pip3  install --no-cache-dir -r /EasyControl/requirements.txt
RUN pip install --no-cache-dir bitsandbytes datasets wandb


ENV ROOT=/EasyControl

FROM base as base_ready
RUN rm -rf /root/.cache/pip/*
# Finalise app setup
WORKDIR ${ROOT}
EXPOSE 8188
# Required for Python print statements to appear in logs
ENV PYTHONUNBUFFERED=1
# Force variant layers to sync cache by setting --build-arg BUILD_DATE
ARG BUILD_DATE
ENV BUILD_DATE=$BUILD_DATE
RUN echo "$BUILD_DATE" > /build_date.txt

FROM base_ready AS default
RUN echo "DEFAULT" >> /variant.txt
ENV CLI_ARGS=""
CMD tail -f /dev/null