#!/bin/bash

# This script runs the pretraining for the Nanochat model on Nvidia DGX Spark.
# It sets up all necessary environment variables and activates the virtual environment
# to ensure training can run even if the user has logged out and back in.
#
# Usage:
#   ./pretrain.sh              # Use default depth of 20 (~561M parameters)
#   ./pretrain.sh --depth 16   # Use custom depth (16 = ~1B parameters)
#   ./pretrain.sh --depth 24   # Use custom depth (24 = ~2.8B parameters)
#
# Credit: Andrej Karpathy - https://github.com/karpathy/nanochat
#
# Author: Jason Cox
# Date: 2025-10-25
# https://github.com/jasonacox/dgx-spark

# Default configuration
DEPTH=20
DEVICE_BATCH_SIZE=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --depth|-d)
            DEPTH="$2"
            shift 2
            ;;
        --batch-size|-b)
            DEVICE_BATCH_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --depth, -d DEPTH         Model depth (number of layers, default: 20)"
            echo "                            Depth 16 ≈ 450M params (dim=1024), 20 ≈ 561M params (dim=1280), 24 ≈ 881M params (dim=1536)"
            echo "  --batch-size, -b SIZE     Device batch size (default: 32)"
            echo "                            Larger values use more memory but may train faster"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Default 20 layers, batch size 32"
            echo "  $0 --depth 16             # Smaller model (~1.0B parameters)"
            echo "  $0 --depth 24             # Larger model (~2.8B parameters)"
            echo "  $0 --batch-size 64        # Larger batch size (more memory)"
            echo "  $0 -d 20 -b 16            # Custom depth and batch size"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we're in the nanochat directory and navigate to it
if [ -d "nanochat" ]; then
    cd nanochat
fi

if [ ! -f "pyproject.toml" ] || [ ! -d ".venv" ]; then
    echo "Error: This script must be run from the dgx-spark/nanochat directory."
    echo "Make sure you've run prepare.sh first."
    exit 1
fi

# Setup CUDA environment variables for Grace Blackwell GB10
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Ensure Rust/Cargo is available
export PATH="$HOME/.cargo/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true

# Activate the Python virtual environment
echo "Activating Python virtual environment..."
source .venv/bin/activate

# Verify PyTorch has CUDA support
echo "Verifying PyTorch CUDA support..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

if [[ ! "$TORCH_VERSION" =~ cu130 ]]; then
    echo "Error: PyTorch is not installed with CUDA 13.0 support."
    echo "Detected PyTorch version: $TORCH_VERSION"
    echo ""
    echo "Please reinstall with CUDA support:"
    echo "  cd ~/dgx-spark/nanochat"
    echo "  rm -rf nanochat/.venv nanochat"
    echo "  ./prepare.sh"
    exit 1
fi

if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "Error: PyTorch cannot detect CUDA devices."
    echo "PyTorch version: $TORCH_VERSION"
    echo "CUDA available: $CUDA_AVAILABLE"
    exit 1
fi

echo "PyTorch $TORCH_VERSION with CUDA support verified ✓"

# Verify CUDA installation
echo "Verifying CUDA installation..."
nvcc --version
if [ $? -ne 0 ]; then
    echo "Error: CUDA not found. Please ensure CUDA 13.0 is properly installed."
    exit 1
fi

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "Error: nvidia-smi failed. Please check GPU drivers and installation."
    exit 1
fi

# Verify wandb is configured
echo "Checking wandb configuration..."
wandb status
if [ $? -ne 0 ]; then
    echo "Warning: wandb not configured. You may need to run 'wandb login' first."
    echo "Continuing anyway..."
fi

# Set optimized settings for DGX Spark GB10
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

echo "Starting pretraining on DGX Spark Grace Blackwell GB10..."
echo "Configuration:"
echo "  - Model depth: $DEPTH layers"
echo "  - Device batch size: $DEVICE_BATCH_SIZE"
echo "  - Training optimized for single GB10 GPU"
echo "  - Using unified memory architecture"
echo ""

# Run pretraining with DGX Spark optimized settings
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=$DEPTH \
    --run="nanochat-pretrain" \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --sample_every=100

echo ""
echo "Pretraining complete!"
echo "Models can be found in: ~/.cache/nanochat/base_checkpoints/"
echo ""
echo "Next step: Run midtraining (fine-tuning) for conversational AI capabilities!"
echo "Run: ./midtrain.sh"
echo ""
echo "Or to chat with your current model:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Start the web interface: python -m scripts.chat_web"
echo "  3. Open the displayed URL in your browser"