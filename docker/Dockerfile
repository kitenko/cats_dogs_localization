FROM python:3.8.5
LABEL maintainer="Andre"

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /detection
COPY file_processing.py onnx_model.onnx requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir
