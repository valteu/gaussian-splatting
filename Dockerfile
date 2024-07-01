FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    echo "nameserver 8.8.4.4" >> /etc/resolv.conf

# Install necessary system packages, COLMAP, and dependencies
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    wget \
    git \
    vim \
    build-essential \
    cmake \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-serialization-dev \
    libglew-dev \
    qtbase5-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libpng-dev \
    libjpeg-dev \
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Create and activate conda environment
RUN conda create -y --name gaussian_splatting python=3.8 && \
    conda init bash && \
    echo "conda activate gaussian_splatting" >> ~/.bashrc

# Set the working directory
WORKDIR /workspace

# Clone the repository with submodules
RUN git clone --recursive https://github.com/valteu/gaussian-splatting.git .

# Install Python dependencies
RUN /bin/bash -c "source activate gaussian_splatting"

# Set the entrypoint to activate conda environment
ENTRYPOINT ["/bin/bash", "-c", "source activate gaussian_splatting && exec bash"]
