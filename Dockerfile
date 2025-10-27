# Use CUDA 12.1 base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/root/.local/bin:$PATH \
    TZ=Asia/Tokyo

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install all system dependencies
RUN apt-get update && apt-get install -y \
    autoconf \
    autoconf-archive \
    automake \
    bison \
    build-essential \
    ca-certificates \
    cmake \
    cmake-curses-gui \
    cmake-gui \
    curl \
    dpkg-dev \
    flex \
    freeglut3-dev \
    g++-10 \
    gcc-10 \
    gettext \
    git \
    gperf \
    gpg \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-iostreams-dev \
    libboost-program-options-dev \
    libboost-serialization-dev \
    libboost-system-dev \
    libceres-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libdbus-1-3 \
    libegl1 \
    libeigen3-dev \
    libflann-dev \
    libfontconfig1 \
    libfreeimage-dev \
    libfreetype6 \
    libgflags-dev \
    libgl1 \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libgles2-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libglib2.0-0 \
    libglu1-mesa-dev \
    libgomp1 \
    libgoogle-glog-dev \
    libgtest-dev \
    libjpeg-dev \
    libmetis-dev \
    libnanoflann-dev \
    libopengl0 \
    libopencv-dev \
    libpng-dev \
    libqt5opengl5-dev \
    libsm6 \
    libsqlite3-dev \
    libsuitesparse-dev \
    libtbb-dev \
    libtiff-dev \
    libtool \
    libx11-6 \
    libx11-dev \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxcb1 \
    libxext-dev \
    libxext6 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libxrender-dev \
    libxrender1 \
    libxt-dev \
    m4 \
    meson \
    ninja-build \
    pkg-config \
    python3-pip \
    python3.10 \
    qtbase5-dev \
    tar \
    unzip \
    wget \
    zip \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMake from Kitware repository
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | \
    gpg --dearmor | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
    https://apt.kitware.com/ubuntu/ $(. /etc/os-release; echo $VERSION_CODENAME) main" \
    > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Build and install OpenMVS
RUN git clone https://github.com/cdcseacave/VCG.git /tmp/vcglib && \
    git clone --recurse-submodules -b v2.3.0 https://github.com/cdcseacave/openMVS.git /tmp/openMVS && \
    cd /tmp/openMVS && \
    mkdir -p make && \
    cd make && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DVCG_ROOT="/tmp/vcglib" \
        -DCUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so \
        -DOpenMVS_USE_CUDA=ON \
        -DCMAKE_POLICY_DEFAULT_CMP0146=OLD && \
    make -j32 && \
    make install && \
    cd / && \
    rm -rf /tmp/openMVS && \
    rm -rf /tmp/vcglib

# Build and Install COLMAP
# NOTE: CUDA architecture 7.5 is for NVIDIA Quadro RTX 8000, please see https://www.hpc.co.jp/library/commentary/gpu-compute-capability-summary/
RUN git clone https://github.com/colmap/colmap.git -b 3.9 /tmp/colmap && \
    cd /tmp/colmap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 && \
    ninja && \
    ninja install && \
    cd / && \
    rm -rf /tmp/colmap

# Add OpenMVS binaries & COLMAP to PATH
ENV PATH="/usr/local/bin/OpenMVS:${PATH}"
ENV PATH="/usr/local/bin/colmap:${PATH}"

# Add PATH settings to .bashrc for interactive shells
RUN echo 'export PATH="/usr/local/bin/OpenMVS:${PATH}"' >> /root/.bashrc && \
    echo 'export PATH="/usr/local/bin/colmap:${PATH}"' >> /root/.bashrc

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /workspace

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install PyTorch and torchvision first
RUN /root/.local/bin/uv pip install --system \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Set default command to bash
CMD ["/bin/bash"]
