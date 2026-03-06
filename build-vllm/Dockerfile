# Build vLLM from Source using Nvidia Container
#
# Credit: @vince_du_66 and @eugr
#     - https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862/60?u=jason188

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# Install essentials
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip git wget patch \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create virtual env
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch + CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install pre-release deps
RUN pip install xgrammar triton flashinfer-python --pre

# Clone vLLM
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /app/vllm

RUN python3 use_existing_torch.py

RUN pip install -r requirements/build.txt

# Apply patch
COPY vllm_patch.diff .
RUN patch -p1 < vllm_patch.diff

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set essential environment variables
ENV TORCH_CUDA_ARCH_LIST=12.0f
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=/app/tiktoken_encodings

# Install vLLM with local build
RUN pip install --no-build-isolation -e . -v --pre

# Download tiktoken encodings
WORKDIR /app
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"


WORKDIR /app/vllm

RUN pip install vllm[audio]

# Expose port
EXPOSE 8888

ENTRYPOINT []

