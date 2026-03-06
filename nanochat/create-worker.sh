#!/bin/bash

# Script to set up a worker node for distributed training with Nanochat
# This script prepares a remote node to be used with pretrain-worker.sh and midtrain-worker.sh
#
# Usage:
#   ./create-worker.sh <worker_hostname>
#
# Examples:
#   ./create-worker.sh dale-ib
#   ./create-worker.sh 192.168.100.5
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
    echo -e "${BLUE}[SETUP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[SETUP]${NC} $1"
}

print_error() {
    echo -e "${RED}[SETUP]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <worker_hostname>"
    echo ""
    echo "Set up a worker node for distributed training with Nanochat."
    echo ""
    echo "Arguments:"
    echo "  worker_hostname    Hostname or IP address of the worker node (required)"
    echo ""
    echo "Examples:"
    echo "  $0 dale-ib                         # Set up worker at dale-ib"
    echo "  $0 192.168.100.5                   # Set up worker at IP address"
    echo ""
    echo "Prerequisites:"
    echo "  1. SSH access with password-less authentication (ssh keys)"
    echo "  2. Worker node has same username as master"
    echo "  3. Worker node has completed base system setup"
    echo "  4. Master node has completed setup.sh and prepare.sh"
    echo ""
    echo "This script will:"
    echo "  - Verify SSH connectivity"
    echo "  - Create dgx-spark directory structure on worker"
    echo "  - Sync all dgx-spark code and scripts"
    echo "  - Sync training cache directories"
    echo "  - Verify Python environment setup"
    exit 1
}

# Check for help flag
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
fi

# Check if worker hostname provided
if [ $# -lt 1 ]; then
    print_error "Missing required argument: worker_hostname"
    echo ""
    show_usage
fi

WORKER_HOST="$1"
WORKER_USER="$USER"
SOURCE_ROOT="$HOME/dgx-spark"
DEST_ROOT="$HOME/dgx-spark"

print_status "==================================================="
print_status "Setting up worker node for distributed training"
print_status "==================================================="
echo ""
print_status "Master: $HOSTNAME ($(whoami))"
print_status "Worker: $WORKER_HOST ($WORKER_USER)"
echo ""

# Step 1: Test SSH connectivity
print_status "Step 1/6: Testing SSH connectivity..."
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${WORKER_USER}@${WORKER_HOST}" "echo 'SSH OK'" > /dev/null 2>&1; then
    print_error "Cannot connect to ${WORKER_USER}@${WORKER_HOST}"
    print_error ""
    print_error "Please ensure:"
    print_error "  1. Worker node is reachable on the network"
    print_error "  2. SSH keys are set up (run: ssh-copy-id ${WORKER_USER}@${WORKER_HOST})"
    print_error "  3. You can manually SSH without a password: ssh ${WORKER_USER}@${WORKER_HOST}"
    exit 1
fi
print_success "SSH connectivity verified"
echo ""

# Step 2: Create directory structure on worker
print_status "Step 2/6: Creating directory structure on worker..."
ssh "${WORKER_USER}@${WORKER_HOST}" "mkdir -p ${DEST_ROOT}/nanochat ~/.cache/nanochat ~/.cache/huggingface" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Directory structure created"
else
    print_error "Failed to create directories on worker"
    exit 1
fi
echo ""

# Step 3: Sync dgx-spark project files
print_status "Step 3/6: Syncing dgx-spark project to worker..."
print_status "This may take a few minutes..."
echo ""

rsync -av --progress \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'wandb/' \
    --exclude 'ib_env.export' \
    "${SOURCE_ROOT}/" \
    "${WORKER_USER}@${WORKER_HOST}:${DEST_ROOT}/"

if [ $? -eq 0 ]; then
    print_success "Project files synced successfully"
else
    print_error "Failed to sync project files"
    exit 1
fi
echo ""

# Step 4: Sync nanochat cache
print_status "Step 4/6: Syncing nanochat cache (checkpoints, datasets)..."
print_status "This may take several minutes depending on checkpoint sizes..."
echo ""

if [ -d "$HOME/.cache/nanochat" ]; then
    rsync -av --progress \
        "$HOME/.cache/nanochat/" \
        "${WORKER_USER}@${WORKER_HOST}:$HOME/.cache/nanochat/"
    
    if [ $? -eq 0 ]; then
        print_success "Nanochat cache synced successfully"
    else
        print_warning "Some files in nanochat cache may not have synced"
    fi
else
    print_warning "No nanochat cache found on master (this is okay for new setups)"
fi
echo ""

# Step 5: Sync HuggingFace cache
print_status "Step 5/6: Syncing HuggingFace datasets cache..."
print_status "This may take several minutes..."
echo ""

if [ -d "$HOME/.cache/huggingface" ]; then
    # First ensure proper permissions on worker
    ssh "${WORKER_USER}@${WORKER_HOST}" "rm -rf $HOME/.cache/huggingface && mkdir -p $HOME/.cache/huggingface" 2>/dev/null
    
    rsync -av --progress \
        "$HOME/.cache/huggingface/" \
        "${WORKER_USER}@${WORKER_HOST}:$HOME/.cache/huggingface/"
    
    if [ $? -eq 0 ]; then
        print_success "HuggingFace cache synced successfully"
    else
        print_warning "Some files in HuggingFace cache may not have synced"
    fi
else
    print_warning "No HuggingFace cache found on master (datasets will download on first use)"
fi
echo ""

# Step 6: Verify environment setup on worker
print_status "Step 6/6: Verifying Python environment on worker..."

VENV_EXISTS=$(ssh "${WORKER_USER}@${WORKER_HOST}" "[ -d ${DEST_ROOT}/nanochat/nanochat/.venv ] && echo 'yes' || echo 'no'" 2>/dev/null)

if [ "$VENV_EXISTS" = "yes" ]; then
    print_success "Python virtual environment found on worker"
    print_status "Environment appears to be ready"
else
    print_warning "Python virtual environment not found on worker"
    print_warning "You will need to run setup on the worker node:"
    print_warning "  ssh ${WORKER_USER}@${WORKER_HOST}"
    print_warning "  cd ${DEST_ROOT}/nanochat"
    print_warning "  ./setup.sh"
    print_warning "  ./prepare.sh"
fi
echo ""

# Final summary
print_success "==================================================="
print_success "Worker node setup complete!"
print_success "==================================================="
echo ""
print_status "Worker node: ${WORKER_HOST}"
print_status "Project directory: ${DEST_ROOT}"
echo ""
print_status "Next steps:"
echo ""
print_status "1. If virtual environment doesn't exist on worker, set it up:"
print_status "   ssh ${WORKER_USER}@${WORKER_HOST}"
print_status "   cd ${DEST_ROOT}/nanochat"
print_status "   ./setup.sh && ./prepare.sh"
echo ""
print_status "2. Start distributed pretraining:"
print_status "   On master: ./pretrain-master.sh ${WORKER_HOST}"
print_status "   On worker: ./pretrain-worker.sh <master-ip>"
echo ""
print_status "3. Or start distributed midtraining:"
print_status "   On master: ./midtrain-master.sh ${WORKER_HOST}"
print_status "   On worker: ./midtrain-worker.sh <master-ip>"
echo ""
print_status "Note: Make sure to start the master script first, then the worker!"
