# DGX Spark Setup Guide

This guide provides comprehensive instructions for setting up the Nvidia DGX Spark AI Supercomputer for LLM training, inference, and Jupyter notebook experimentation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Environment Configuration](#environment-configuration)
- [LLM Training Setup](#llm-training-setup)
- [Inference Configuration](#inference-configuration)
- [Jupyter Notebook Setup](#jupyter-notebook-setup)
- [Quick Start Examples](#quick-start-examples)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before setting up your DGX Spark system, ensure you have:

- Access to the DGX Spark system with appropriate user permissions
- SSH access configured to the DGX Spark system
- Basic familiarity with Linux command line
- Understanding of Python environments and package management

## Initial Setup

### 1. System Access

Connect to your DGX Spark system:

```bash
ssh your-username@dgx-spark-hostname
```

### 2. Verify GPU Access

Check available GPUs on the system:

```bash
nvidia-smi
```

This should display all available NVIDIA GPUs with their current utilization and memory usage.

### 3. Check CUDA Installation

Verify CUDA is properly installed:

```bash
nvcc --version
```

### 4. Verify Docker/Container Runtime

DGX systems typically use NVIDIA Container Runtime:

```bash
docker --version
nvidia-container-toolkit --version
```

## Environment Configuration

### Setting Up Python Environment

We recommend using conda or virtual environments for managing dependencies:

#### Using Conda

```bash
# Download and install Miniconda (if not already available)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create a new environment for LLM work
conda create -n llm-env python=3.10
conda activate llm-env
```

#### Using venv

```bash
python3 -m venv ~/llm-env
source ~/llm-env/bin/activate
```

### Install Essential Packages

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install jupyter jupyterlab ipywidgets
```

## LLM Training Setup

### 1. Install Training Frameworks

```bash
# Install additional training dependencies
pip install deepspeed
pip install transformers[torch]
pip install accelerate bitsandbytes
pip install wandb tensorboard  # For experiment tracking
```

### 2. Configure Multi-GPU Training

Create a DeepSpeed configuration file (`ds_config.json`):

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

### 3. Example Training Script

```bash
# Run distributed training across all available GPUs
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir ./output \
    --deepspeed ds_config.json
```

### 4. Monitor Training

Use NVIDIA tools to monitor GPU utilization:

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop for a better interface
nvtop
```

## Inference Configuration

### 1. Install Inference Dependencies

```bash
pip install vllm
pip install triton
pip install text-generation-inference
```

### 2. Using vLLM for Fast Inference

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="meta-llama/Llama-2-7b-hf", 
          tensor_parallel_size=4)  # Use 4 GPUs

# Generate text
prompts = ["Hello, my name is", "The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

### 3. Model Serving with TGI

```bash
# Launch Text Generation Inference server
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-shard 4
```

## Jupyter Notebook Setup

### 1. Install JupyterLab

```bash
pip install jupyterlab ipywidgets ipykernel
pip install jupyter-ai  # Optional: AI-powered Jupyter assistant
```

### 2. Configure Jupyter for Remote Access

Generate Jupyter configuration:

```bash
jupyter lab --generate-config
```

Set up password protection:

```bash
jupyter lab password
```

### 3. Launch Jupyter Lab

For local access:

```bash
jupyter lab --port=8888
```

For remote access (accessible from other machines):

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 4. Access Jupyter

Open your browser and navigate to:
- Local: `http://localhost:8888`
- Remote: `http://dgx-spark-hostname:8888`

### 5. GPU Acceleration in Notebooks

Verify GPU access in a notebook:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### 6. Example LLM Experimentation Notebook

Create a new notebook with this starter code:

```python
# Import libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "gpt2"  # Start with a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Quick Start Examples

### Example 1: Fine-tune a Model

```bash
# Clone example repository
git clone https://github.com/huggingface/transformers.git
cd transformers/examples/pytorch/language-modeling

# Run fine-tuning
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./output
```

### Example 2: Run Inference on Custom Prompts

```python
from transformers import pipeline

# Create a text generation pipeline
generator = pipeline('text-generation', 
                    model='gpt2',
                    device=0)  # Use GPU 0

# Generate text
result = generator("Once upon a time", 
                  max_length=50, 
                  num_return_sequences=1)
print(result[0]['generated_text'])
```

### Example 3: Load and Use a Large Language Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import bitsandbytes  # Required for 8-bit quantization

# Load model in 8-bit for memory efficiency
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Chat with the model
def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    # device_map="auto" handles device placement automatically
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = chat("Explain machine learning in simple terms:")
print(response)
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Reduce batch size in your training/inference configuration
2. Use gradient accumulation for effective larger batch sizes
3. Enable mixed precision training (FP16/BF16)
4. Use model parallelism or offloading strategies

```python
# Example: Load model with 8-bit quantization
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)
```

### GPU Not Detected

```bash
# Check if NVIDIA drivers are loaded
lsmod | grep nvidia

# Restart NVIDIA persistence daemon
sudo systemctl restart nvidia-persistenced

# Check NVIDIA container runtime
nvidia-container-cli info
```

### Jupyter Kernel Issues

```bash
# Reinstall kernel
python -m ipykernel install --user --name=llm-env

# List available kernels
jupyter kernelspec list

# Remove old kernels if needed
jupyter kernelspec uninstall unwanted-kernel
```

### Permission Issues

If you encounter permission errors:

```bash
# Add user to docker group (requires logout/login)
sudo usermod -aG docker $USER

# Fix Python package permissions
pip install --user package-name
```

## Additional Resources

- [NVIDIA DGX Documentation](https://docs.nvidia.com/dgx/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Contributing

Helpful tools and notes for the Nvidia DGX Spark AI Supercomputer are welcome. Please submit issues or pull requests with improvements to this guide.

## License

MIT License - See LICENSE file for details
