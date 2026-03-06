#!/bin/bash

# This script starts a CLI chat session with your trained Nanochat model on Nvidia DGX Spark.
# It sets up all necessary environment variables and launches the command-line interface.
#
# Usage:
#   ./chat_cli.sh              # Auto-detect most advanced model
#   ./chat_cli.sh --source rl  # Use RL model
#   ./chat_cli.sh --source sft # Use SFT model
#   ./chat_cli.sh --source mid # Use midtrain model
#   ./chat_cli.sh --source base # Use base model
#
# Credit: Andrej Karpathy - https://github.com/karpathy/nanochat
#
# Author: Jason Cox
# Date: 2025-11-05
# https://github.com/jasonacox/dgx-spark

# Parse command line arguments
SPECIFIED_SOURCE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --source|-s)
            SPECIFIED_SOURCE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--source MODEL]"
            echo ""
            echo "Options:"
            echo "  --source, -s MODEL   Specify model source: rl, sft, mid, or base"
            echo "                       (default: auto-detect most advanced)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                   # Auto-detect most advanced model"
            echo "  $0 --source rl       # Use RL model"
            echo "  $0 --source sft      # Use SFT model"
            echo "  $0 --source mid      # Use midtrain model"
            echo "  $0 --source base     # Use base pretrain model"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we're in the nanochat directory
cd nanochat
if [ ! -f "pyproject.toml" ] || [ ! -d ".venv" ]; then
    echo "Error: This script must be run from the nanochat directory."
    echo "Make sure you've run setup.sh first and are in the nanochat folder."
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

# Verify CUDA installation
echo "Verifying CUDA installation..."
nvcc --version
if [ $? -ne 0 ]; then
    echo "Error: CUDA not found. Please ensure CUDA 13.0 is properly installed."
    exit 1
fi

# Check for trained models
echo "Checking for trained models..."
MODEL_SOURCE=""

# If user specified a source, validate and use it
if [ -n "$SPECIFIED_SOURCE" ]; then
    case "$SPECIFIED_SOURCE" in
        rl)
            if [ -d "$HOME/.cache/nanochat/chatrl_checkpoints" ]; then
                MODEL_SOURCE="rl"
                echo "Using specified RL (Reinforcement Learning) model"
            else
                echo "Error: RL checkpoints not found at $HOME/.cache/nanochat/chatrl_checkpoints"
                exit 1
            fi
            ;;
        sft)
            if [ -d "$HOME/.cache/nanochat/chatsft_checkpoints" ]; then
                MODEL_SOURCE="sft"
                echo "Using specified SFT (Supervised Fine-tuning) model"
            else
                echo "Error: SFT checkpoints not found at $HOME/.cache/nanochat/chatsft_checkpoints"
                exit 1
            fi
            ;;
        mid)
            if [ -d "$HOME/.cache/nanochat/mid_checkpoints" ]; then
                MODEL_SOURCE="mid"
                echo "Using specified midtraining model"
            else
                echo "Error: Midtraining checkpoints not found at $HOME/.cache/nanochat/mid_checkpoints"
                exit 1
            fi
            ;;
        base)
            if [ -d "$HOME/.cache/nanochat/base_checkpoints" ]; then
                MODEL_SOURCE="base"
                echo "Using specified base pretraining model"
            else
                echo "Error: Base checkpoints not found at $HOME/.cache/nanochat/base_checkpoints"
                exit 1
            fi
            ;;
        *)
            echo "Error: Invalid model source '$SPECIFIED_SOURCE'"
            echo "Valid options: rl, sft, mid, base"
            exit 1
            ;;
    esac
else
    # Auto-detect most advanced model
    if [ -d "$HOME/.cache/nanochat/chatrl_checkpoints" ]; then
        MODEL_SOURCE="rl"
        echo "Auto-detected RL (Reinforcement Learning) checkpoints - using most advanced model"
    elif [ -d "$HOME/.cache/nanochat/chatsft_checkpoints" ]; then
        MODEL_SOURCE="sft"
        echo "Auto-detected SFT (Supervised Fine-tuning) checkpoints"
    elif [ -d "$HOME/.cache/nanochat/mid_checkpoints" ]; then
        MODEL_SOURCE="mid"
        echo "Auto-detected midtraining checkpoints"
    elif [ -d "$HOME/.cache/nanochat/base_checkpoints" ]; then
        MODEL_SOURCE="base"
        echo "Auto-detected base pretraining checkpoints"
    else
        echo "Error: No trained models found."
        echo "Please complete at least pretraining first:"
        echo "  ./pretrain.sh"
        exit 1
    fi
fi

# Set optimized settings for DGX Spark GB10
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Detect model size from checkpoint metadata
CHECKPOINT_DIR=""
case "$MODEL_SOURCE" in
    rl) CHECKPOINT_DIR="$HOME/.cache/nanochat/chatrl_checkpoints" ;;
    sft) CHECKPOINT_DIR="$HOME/.cache/nanochat/chatsft_checkpoints" ;;
    mid) CHECKPOINT_DIR="$HOME/.cache/nanochat/mid_checkpoints" ;;
    base) CHECKPOINT_DIR="$HOME/.cache/nanochat/base_checkpoints" ;;
esac

# Find a meta file to extract model info
META_FILE=$(find "$CHECKPOINT_DIR" -name "meta_*.json" -type f | head -1)
if [ -n "$META_FILE" ]; then
    MODEL_INFO=$(python3 << PYTHON_EOF
import json
try:
    with open('$META_FILE', 'r') as f:
        meta = json.load(f)
    model_config = meta.get('model_config', {})
    n_layer = model_config.get('n_layer', 20)
    n_embd = model_config.get('n_embd', 1280)
    
    # Calculate parameters (same formula as hf_prepare.sh)
    vocab_size = model_config.get('vocab_size', 65536)
    token_embedding_params = vocab_size * n_embd
    output_head_params = vocab_size * n_embd
    params_per_layer = 12 * n_embd * n_embd  # 4x for attention + 8x for MLP
    total_params = token_embedding_params + (n_layer * params_per_layer) + output_head_params
    
    params_millions = total_params / 1_000_000
    if params_millions >= 1000:
        params_billions = total_params / 1_000_000_000
        if params_billions >= 10:
            params_display = f"{int(params_billions)}B"
        else:
            params_display = f"{params_billions:.1f}B"
    else:
        params_display = f"{int(params_millions)}M"
    
    print(f"{params_display}|{n_layer}|{n_embd}")
except:
    print("561M|20|1280")
PYTHON_EOF
)
    PARAM_COUNT=$(echo "$MODEL_INFO" | cut -d'|' -f1)
    N_LAYERS=$(echo "$MODEL_INFO" | cut -d'|' -f2)
    N_EMBD=$(echo "$MODEL_INFO" | cut -d'|' -f3)
else
    PARAM_COUNT="561M"
    N_LAYERS="20"
    N_EMBD="1280"
fi

echo ""
echo "Starting Nanochat CLI on DGX Spark..."
echo ""
echo "🤖 Your personal ChatGPT-like AI is starting up!"
echo ""
echo "Features of your trained model:"
echo "  ✅ ${PARAM_COUNT} parameters (${N_LAYERS} layers, ${N_EMBD} dim) trained from scratch"
echo "  ✅ Optimized for DGX Spark's Grace Blackwell architecture"
echo "  ✅ Fully yours - no API dependencies"
echo "  ✅ Privacy-focused - runs locally on your hardware"
echo ""
echo "Commands:"
echo "  • Type your message and press Enter to chat"
echo "  • Type 'exit' or 'quit' to end the session"
echo "  • Press Ctrl+C to interrupt"
echo ""
echo "Note: As a micro-model, it may occasionally:"
echo "  ⚠️  Make factual errors or hallucinate"
echo "  ⚠️  Be naive or childlike in responses"
echo "  ⚠️  Have limitations compared to large commercial models"
echo ""
echo "Loading model..."

# Launch the CLI chat interface with the appropriate model source
python -m scripts.chat_cli --source "$MODEL_SOURCE"

echo ""
echo "Chat session ended."
echo "To start another session, run: ./chat_cli.sh"
