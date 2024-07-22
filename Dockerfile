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
    ninja-build \
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

# Install PyTorch with CUDA 11.8 support
RUN /bin/bash -c "source activate gaussian_splatting && \
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113"

# Set the working directory
WORKDIR /workspace

# Clone the repository with submodules
RUN git clone --recursive https://github.com/valteu/gaussian-splatting.git .

# Set CUDA architecture flags
ENV TORCH_CUDA_ARCH_LIST="7.0+PTX"

# Install Python dependencies and submodules
RUN /bin/bash -c "source activate gaussian_splatting && \
    cd submodules/diff-gaussian-rasterization/ && \
    pip install -e . && \
    cd ../simple-knn && \
    pip install -e . && \
    cd ../.."

# Set the entrypoint to activate conda environment
ENTRYPOINT ["/bin/bash", "-c", "source activate gaussian_splatting && exec bash"]
