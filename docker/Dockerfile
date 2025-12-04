FROM hub.rat.dev/nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda

ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV CONDA_DEFAULT_ENV=base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV MUJOCO_GL=egl

COPY docker/sources.list /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        build-essential \
        pkg-config \
        unzip \
        zip \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        libegl1 \
        libxi6 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxext6 \
        libxrender1 \
        mesa-utils \
        mesa-vulkan-drivers \
        vulkan-tools \
        patchelf && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.pip
COPY docker/pip.conf /root/.pip/pip.conf
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN git config --global http.version HTTP/1.1 && \
    git config --global http.postBuffer 524288000

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -afy

RUN ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN ${CONDA_DIR}/bin/conda create -y -n uvd python=3.9 && \
    ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/envs/uvd/bin:${CONDA_DIR}/bin:${PATH}
ENV CONDA_DEFAULT_ENV=uvd
ENV PIP_NO_BUILD_ISOLATION=1

WORKDIR /workspace

COPY README.md requirements.txt setup.py /workspace/
COPY demo.py /workspace/demo.py
COPY uvd /workspace/uvd
COPY utils /workspace/utils
COPY datasets /workspace/datasets
COPY scripts /workspace/scripts

# Install standalone visual foundation models referenced in README (always on)
RUN set -eux; 
RUN ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir "git+https://github.com/facebookresearch/vip.git"; \
    ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir "git+https://github.com/facebookresearch/r3m.git";
    
RUN ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir "git+https://github.com/penn-pal-lab/LIV.git"; \
    ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir "git+https://github.com/penn-pal-lab/LIV.git#subdirectory=liv/models/clip";

RUN ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir "git+https://github.com/facebookresearch/eai-vc.git#subdirectory=vc_models"

RUN ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir "pip<24" "setuptools==59.5.0" "wheel==0.38.4" && \
    ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir -r requirements.txt && \
    ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir -e .

RUN ${CONDA_DIR}/bin/conda run -n uvd pip install --no-cache-dir httpx[socks] glog

CMD ["/bin/bash"]
