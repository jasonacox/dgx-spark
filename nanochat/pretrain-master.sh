#!/bin/bash

# This script runs distributed pretraining for the Nanochat model on the MASTER node
# in a two-node DGX Spark cluster connected via InfiniBand.
# This script should be run on the master node (NODE_RANK=0).
#
# Usage:
#   ./pretrain-master.sh [worker_host] [options]
#   
# Examples:
#   ./pretrain-master.sh spark-node02
#   ./pretrain-master.sh 192.168.100.5 --depth 24
#   ./pretrain-master.sh spark-worker --depth 16 --batch-size 16
#
# Credit: 
#   - Andrej Karpathy - https://github.com/karpathy/nanochat
#   - Emaad Manzoor - Multi-node setup guide
#
# Author: Jason Cox
# Date: 2025-12-23
# https://github.com/jasonacox/dgx-spark

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[MASTER]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[MASTER]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[MASTER]${NC} $1"
}

print_error() {
    echo -e "${RED}[MASTER]${NC} $1"
}

# Default configuration
DEPTH=20
DEVICE_BATCH_SIZE=32
WORKER_HOST=""
MASTER_PORT=29500

# Function to show usage
show_usage() {
    echo "Usage: $0 <worker_host> [options]"
    echo ""
    echo "Run distributed pretraining on MASTER node of a two-node DGX Spark cluster."
    echo ""
    echo "Arguments:"
    echo "  worker_host               Hostname or IP of the worker node (required)"
    echo ""
    echo "Options:"
    echo "  --depth, -d DEPTH         Model depth (number of layers, default: 20)"
    echo "                            Depth 16 ≈ 1.0B params, 20 ≈ 1.9B params, 24 ≈ 2.8B params"
    echo "  --batch-size, -b SIZE     Device batch size (default: 32)"
    echo "                            Larger values use more memory but may train faster"
    echo "  --port, -p PORT           Master communication port (default: 29500)"
    echo "  --help, -h                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 spark-node02                    # Connect to worker at spark-node02"
    echo "  $0 192.168.100.5 --depth 16        # Custom worker IP and smaller model"
    echo "  $0 spark-worker -d 24 -b 16        # Custom depth and batch size"
    echo ""
    echo "Prerequisites:"
    echo "  1. Both nodes must have completed setup.sh and prepare.sh"
    echo "  2. InfiniBand connection must be working between nodes"
    echo "  3. SSH key authentication must be set up"
    echo "  4. Run pretrain-worker.sh on the worker node AFTER starting this script"
    exit 1
}

# Check for help flag first
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        show_usage
    fi
done

# Parse command line arguments
if [ $# -lt 1 ]; then
    print_error "Missing required argument: worker_host"
    echo ""
    show_usage
fi

WORKER_HOST="$1"
shift

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
        --port|-p)
            MASTER_PORT="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_status "DGX Spark Distributed Training - MASTER NODE"
print_status "=============================================="
echo ""

# Check if we're in the nanochat directory and navigate to it
if [ -d "nanochat" ]; then
    cd nanochat
fi

if [ ! -f "pyproject.toml" ] || [ ! -d ".venv" ]; then
    print_error "This script must be run from the dgx-spark/nanochat directory."
    print_error "Make sure you've run prepare.sh first."
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
print_status "Activating Python virtual environment..."
source .venv/bin/activate

# Set up distributed training environment
print_status "Setting up distributed training environment..."

# Get InfiniBand interface
NODE_RANK=0
IB_IF=$(/usr/sbin/ibdev2netdev | awk '/(Up|ACTIVE)/{print $5; exit}')
if [ -z "$IB_IF" ]; then
    print_warning "Could not detect InfiniBand interface, falling back to primary network interface"
    IB_IF=$(ip route | grep default | awk '{print $5}' | head -1)
fi

# Get master address from IB interface
MASTER_ADDR=$(ip -o -4 addr show dev "$IB_IF" | awk '{print $4}' | cut -d/ -f1)
if [ -z "$MASTER_ADDR" ]; then
    print_error "Could not determine MASTER_ADDR from interface $IB_IF"
    exit 1
fi

# Create environment export file
cat > ib_env.export <<EOF
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$NODE_RANK
export WORLD_SIZE=2
export NCCL_SOCKET_IFNAME=$IB_IF
EOF

print_status "Distributed training configuration:"
print_status "  Master Address: $MASTER_ADDR"
print_status "  Master Port: $MASTER_PORT"
print_status "  Node Rank: $NODE_RANK (MASTER)"
print_status "  World Size: 2"
print_status "  Network Interface: $IB_IF"
print_status "  Worker Host: $WORKER_HOST"
print_status "  Model Depth: $DEPTH layers"
print_status "  Device Batch Size: $DEVICE_BATCH_SIZE"
echo ""

# Test connectivity to worker node
print_status "Testing connectivity to worker node..."
if ! ping -c 1 -W 3 "$WORKER_HOST" > /dev/null 2>&1; then
    print_warning "Cannot ping worker host $WORKER_HOST"
    print_warning "Training may fail if worker is not reachable"
fi

# Verify PyTorch has CUDA support
print_status "Verifying PyTorch CUDA support..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

if [[ ! "$TORCH_VERSION" =~ cu130 ]]; then
    print_error "PyTorch is not installed with CUDA 13.0 support."
    print_error "Detected PyTorch version: $TORCH_VERSION"
    exit 1
fi

if [ "$CUDA_AVAILABLE" != "True" ]; then
    print_error "PyTorch cannot detect CUDA devices."
    exit 1
fi

print_success "PyTorch $TORCH_VERSION with CUDA support verified"

# Verify CUDA installation
print_status "Verifying CUDA installation..."
nvcc --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_error "CUDA not found. Please ensure CUDA 13.0 is properly installed."
    exit 1
fi

# Verify GPU availability
print_status "Checking GPU availability..."
nvidia-smi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_error "nvidia-smi failed. Please check GPU drivers and installation."
    exit 1
fi

# Check wandb configuration
print_status "Checking wandb configuration..."
wandb status > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_warning "wandb not configured. You may need to run 'wandb login' first."
    print_warning "Continuing anyway..."
fi

# Set optimized settings for DGX Spark GB10
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Source the distributed environment
source ib_env.export

print_success "Environment setup complete!"
echo ""
print_status "IMPORTANT: Now start the worker node by running on $WORKER_HOST:"
print_status "  ./pretrain-worker.sh $MASTER_ADDR --depth $DEPTH --batch-size $DEVICE_BATCH_SIZE --port $MASTER_PORT"
echo ""
print_status "Waiting 10 seconds for you to start the worker node..."
print_status "(You can press Ctrl+C to cancel and start manually)"

for i in {10..1}; do
    echo -n -e "\r${BLUE}[MASTER]${NC} Starting training in $i seconds... "
    sleep 1
done
echo ""

print_status "Starting distributed pretraining on DGX Spark cluster..."
print_status "This is the MASTER node - training will begin when worker connects"
echo ""

# Run distributed pretraining with DGX Spark optimized settings
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=${NODE_RANK} \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m scripts.base_train -- \
    --depth=$DEPTH \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --sample_every=100 \
    --save_every=1000 \
    --run="nanochat-2spark-pretrain"

echo ""
if [ $? -eq 0 ]; then
    print_success "Distributed pretraining completed successfully!"
    print_status "Models can be found in: ~/.cache/nanochat/base_checkpoints/"
    echo ""
    print_status "Next steps:"
    print_status "  1. Run distributed midtraining: ./midtrain-master.sh $WORKER_HOST"
    print_status "  2. Or test your model: source .venv/bin/activate && python -m scripts.chat_web"
else
    print_error "Distributed pretraining failed!"
    print_error "Check the logs above for error details"
    exit 1
fi