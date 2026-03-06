# Build a Docker image for vLLM
#
# Use the official vLLM Docker image as the base image
# from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags
#
FROM nvcr.io/nvidia/vllm:25.09-py3
#FROM nvcr.io/nvidia/vllm:25.11-py3

WORKDIR /workspace

# Clone vLLM and checkout a specific version
# To pin to a specific release, uncomment and modify the git checkout line:
RUN git clone https://github.com/vllm-project/vllm.git

# The following line checks out a specific commit for reproducibility
# required for Qwen vision model (e.g. Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
RUN cd vllm && git checkout 5bb1da5190b54aefb08478c6b1170f97722b8bdb

# Install vLLM with existing PyTorch
RUN cd vllm && \ 
    python use_existing_torch.py && \
    pip install -r requirements/build.txt && \
    pip install --no-build-isolation -e .

EXPOSE 8000

CMD ["/bin/bash"]
