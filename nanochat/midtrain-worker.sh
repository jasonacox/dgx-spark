#!/bin/bash

# This script runs distributed midtraining (fine-tuning) for the Nanochat model on the WORKER node
# in a two-node DGX Spark cluster connected via InfiniBand.
# This script should be run on the worker node (NODE_RANK=1) AFTER starting the master.
#
# Usage:
#   ./midtrain-worker.sh <master_address> [options]
#   
# Examples:
#   ./midtrain-worker.sh 192.168.100.4
#   ./midtrain-worker.sh 169.254.69.64 --depth 24
#   ./midtrain-worker.sh spark-master --depth 16 --batch-size 16
#
# Credit: 
#   - Andrej Karpathy - https://github.com/karpathy/nanochat
#   - Emaad Manzoor - Multi-node setup guide
#
# Author: Jason Cox
# Date: 2026-01-19
# https://github.com/jasonacox/dgx-spark

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[WORKER]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[WORKER]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WORKER]${NC} $1"
}

print_error() {
    echo -e "${RED}[WORKER]${NC} $1"
}

# Default configuration
DEPTH=20
DEVICE_BATCH_SIZE=32
MASTER_ADDR=""
MASTER_PORT=29501

# Function to show usage
show_usage() {
    echo "Usage: $0 <master_address> [options]"
    echo ""
    echo "Run distributed midtraining (fine-tuning) on WORKER node of a two-node DGX Spark cluster."
    echo ""
    echo "Arguments:"
    echo "  master_address            IP address or hostname of the master node (required)"
    echo ""
    echo "Options:"
    echo "  --depth, -d DEPTH         Model depth (number of layers, default: 20)"
    echo "                            Must match the master node's depth setting"
    echo "  --batch-size, -b SIZE     Device batch size (default: 32)"
    echo "                            Must match the master node's batch size setting"
    echo "  --port, -p PORT           Master communication port (default: 29501)"
    echo "                            Must match the master node's port setting"
    echo "  --help, -h                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 192.168.100.4                   # Connect to master at IP address"
    echo "  $0 spark-master --depth 16         # Connect to named host with custom depth"
    echo "  $0 169.254.69.64 -d 24 -b 16       # Custom depth and batch size"
    echo ""
    echo "Prerequisites:"
    echo "  1. Both nodes must have completed setup.sh and prepare.sh"
    echo "  2. Both nodes should have a pretrained model checkpoint"
    echo "  3. InfiniBand connection must be working between nodes"
    echo "  4. Master node must be running midtrain-master.sh first"
    echo "  5. All training parameters must match between master and worker"
    echo ""
    echo "Note: Start the master node FIRST, then run this worker script!"
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
    print_error "Missing required argument: master_address"
    echo ""
    show_usage
fi

MASTER_ADDR="$1"
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

print_status "DGX Spark Distributed Midtraining - WORKER NODE"
print_status "================================================"
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
NODE_RANK=1
IB_IF=$(/usr/sbin/ibdev2netdev | awk '/(Up|ACTIVE)/{print $5; exit}')
if [ -z "$IB_IF" ]; then
    print_warning "Could not detect InfiniBand interface, falling back to primary network interface"
    IB_IF=$(ip route | grep default | awk '{print $5}' | head -1)
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
print_status "  Node Rank: $NODE_RANK (WORKER)"
print_status "  World Size: 2"
print_status "  Network Interface: $IB_IF"
print_status "  Model Depth: $DEPTH layers"
print_status "  Device Batch Size: $DEVICE_BATCH_SIZE"
echo ""

# Test connectivity to master node
print_status "Testing connectivity to master node..."
if ! ping -c 1 -W 3 "$MASTER_ADDR" > /dev/null 2>&1; then
    print_error "Cannot ping master node at $MASTER_ADDR"
    print_error "Please check:"
    print_error "  1. Master node IP address is correct"
    print_error "  2. Network connectivity between nodes"
    print_error "  3. InfiniBand link is up and configured"
    exit 1
fi
print_success "Master node is reachable"

# Test if master port is listening
print_status "Checking if master node is ready..."
timeout 5 bash -c "echo >/dev/tcp/$MASTER_ADDR/$MASTER_PORT" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Master node is accepting connections on port $MASTER_PORT"
else
    print_warning "Master node may not be ready yet (port $MASTER_PORT not responding)"
    print_warning "Training will wait for master to become available"
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

# Check wandb configuration (optional on worker)
print_status "Checking wandb configuration..."
wandb status > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_warning "wandb not configured on worker node (this is usually fine)"
fi

# Download identity conversations file if it doesn't exist
if [ ! -f "$HOME/.cache/nanochat/identity_conversations.jsonl" ]; then
    print_status "Downloading identity conversations dataset..."
    mkdir -p "$HOME/.cache/nanochat"
    curl -L -o "$HOME/.cache/nanochat/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# Set optimized settings for DGX Spark GB10
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Source the distributed environment
source ib_env.export

print_success "Environment setup complete!"
echo ""
print_status "Connecting to master node for distributed training..."
print_warning "This worker will wait for the master node to start training"
print_warning "You may see temporary connection warnings - this is normal"
echo ""

print_status "Starting distributed midtraining on DGX Spark cluster..."
print_status "This is the WORKER node - waiting for master to coordinate training"
echo ""

# Run distributed midtraining with DGX Spark optimized settings
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=${NODE_RANK} \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m scripts.mid_train -- \
    --run="nanochat-2spark-midtrain"

echo ""
if [ $? -eq 0 ]; then
    print_success "Distributed midtraining completed successfully!"
    print_status "Fine-tuned models can be found in: ~/.cache/nanochat/mid_checkpoints/"
    echo ""
    print_status "Worker node training complete - check master node for next steps"
else
    print_error "Distributed midtraining failed!"
    print_error "Check the logs above for error details"
    print_error "Also check the master node logs"
    exit 1
fi
