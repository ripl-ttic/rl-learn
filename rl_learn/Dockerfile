# Base image
FROM nvidia/cudagl:9.0-runtime-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN git clone https://github.com/ripl-ttic/rl-learn.git
RUN /bin/bash -c "cd rl-learn; conda create --name rl-learn python=3.6 --file requirements.txt"
RUN /bin/bash -c ". activate rl-learn; conda install pytorch torchvision cudatoolkit=9.0 -c pytorch"

RUN . activate rl-learn
RUN apt-get update && apt install -y \
    python3-dev \
    zlib1g-dev \
    libjpeg-dev \
    cmake \
    swig \
    python-pyglet \
    python3-opengl \
    libboost-all-dev \
    libsdl2-dev \
    libosmesa6-dev \
    patchelf \
    ffmpeg \
    xvfb &&\
    rm -rf /var/lib/apt/lists/*
RUN . deactivate

RUN git clone https://github.com/openai/gym.git
RUN /bin/bash -c ". activate rl-learn; cd gym; pip install -e '.[atari]'"

RUN git clone https://github.com/openai/baselines.git
RUN /bin/bash -c ". activate rl-learn; cd baselines; pip install -e ."

RUN pip install tb-nightly