#!/bin/bash

# NanoChat OpenAI-Compatible API Service Launcher
# Provides an OpenAI-compatible chat API endpoint for nanochat models
#
# Usage:
#   ./chat_web.sh [options]
#
# Examples:
#   ./chat_web.sh --source sft --port 8000
#   ./chat_web.sh --source rl --num-gpus 4 --port 8001
#   ./chat_web.sh --help

# Default values
SOURCE=""
PORT=8000
TEMPERATURE=0.8
TOP_K=50
MAX_TOKENS=512
HOST="0.0.0.0"
DTYPE="bfloat16"
RUN_AS_SERVER=false
STOP_SERVER=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --source|-s)
            SOURCE="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --temperature|-t)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-k|-k)
            TOP_K="$2"
            shift 2
            ;;
        --max-tokens|-m)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --dtype|-d)
            DTYPE="$2"
            shift 2
            ;;
        --server)
            RUN_AS_SERVER=true
            shift
            ;;
        --stop)
            STOP_SERVER=true
            shift
            ;;
        --help|-h)
            echo "NanoChat OpenAI-Compatible API Service"
            echo ""
            echo "Usage: ./chat_web.sh [options]"
            echo ""
            echo "Options:"
            echo "  --source, -s         Model source: rl|sft|mid|base (auto-detect if not specified)"
            echo "  --port, -p           Port to run the server on (default: 8000)"
            echo "  --temperature, -t    Default temperature for generation (default: 0.8)"
            echo "  --top-k, -k          Default top-k sampling parameter (default: 50)"
            echo "  --max-tokens, -m     Default max tokens for generation (default: 512)"
            echo "  --host               Host to bind the server to (default: 0.0.0.0)"
            echo "  --dtype, -d          Data type: float32|bfloat16 (default: bfloat16)"
            echo "  --server             Run as background server with logging"
            echo "  --stop               Stop the running server"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./chat_web.sh --source sft --port 8000"
            echo "  ./chat_web.sh --source rl --port 8001 --server"
            echo "  ./chat_web.sh --stop"
            echo "  ./chat_web.sh -s mid -p 8002 -t 0.7"
            echo ""
            echo "After starting, use the OpenAI SDK to connect:"
            echo "  from openai import OpenAI"
            echo "  client = OpenAI(api_key='not-needed', base_url='http://localhost:8000/v1')"
            echo "  response = client.chat.completions.create("
            echo "      model='nanochat',"
            echo "      messages=[{'role': 'user', 'content': 'Hello!'}],"
            echo "      stream=True"
            echo "  )"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './chat_web.sh --help' for usage information"
            exit 1
            ;;
    esac
done

# Handle --stop option
if [ "$STOP_SERVER" = true ]; then
    echo "Stopping chat_web server..."
    pkill -f "python -m scripts.chat_web"
    if [ $? -eq 0 ]; then
        echo "✓ Server stopped successfully"
    else
        echo "No running server found"
    fi
    exit 0
fi

# Check if we're in the nanochat directory and navigate to it
if [ -d "nanochat" ]; then
    cd nanochat
fi

if [ ! -f "pyproject.toml" ] || [ ! -d ".venv" ]; then
    echo "Error: This script must be run from the nanochat parent directory."
    echo "Make sure you've run setup.sh first."
    exit 1
fi

# Check if chat_web.py exists in the nanochat scripts directory
# If not, copy it from the parent dgx-spark repo
CHAT_SERVICE_DEST="scripts/chat_web.py"
CHAT_SERVICE_SOURCE="../chat_web.py"

if [ ! -f "$CHAT_SERVICE_DEST" ]; then
    echo "Installing chat_web.py into nanochat scripts directory..."
    if [ -f "$CHAT_SERVICE_SOURCE" ]; then
        # Create scripts directory if it doesn't exist
        mkdir -p "scripts"
        cp "$CHAT_SERVICE_SOURCE" "$CHAT_SERVICE_DEST"
        echo "✓ chat_web.py installed successfully"
    else
        echo "Error: chat_web.py not found in dgx-spark repo"
        echo "Expected location: $CHAT_SERVICE_SOURCE"
        exit 1
    fi
fi

# Setup CUDA environment variables for Grace Blackwell GB10
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Ensure Rust/Cargo is available
export PATH="$HOME/.cargo/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true

# Set optimized settings for DGX Spark GB10
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Auto-detect model source if not specified
if [ -z "$SOURCE" ]; then
    CACHE_DIR="$HOME/.cache/nanochat"
    
    echo "No --source specified, auto-detecting trained model..."
    
    if [ -d "$CACHE_DIR/chatrl_checkpoints" ]; then
        SOURCE="rl"
        echo "Auto-detected RL model (most advanced)"
    elif [ -d "$CACHE_DIR/chatsft_checkpoints" ]; then
        SOURCE="sft"
        echo "Auto-detected SFT model"
    elif [ -d "$CACHE_DIR/mid_checkpoints" ]; then
        SOURCE="mid"
        echo "Auto-detected midtrained model"
    elif [ -d "$CACHE_DIR/base_checkpoints" ]; then
        SOURCE="base"
        echo "Auto-detected base model"
    else
        echo "Error: No trained model found in $CACHE_DIR"
        echo "Please specify a model source with --source (rl|sft|mid|base)"
        echo "Or train a model first using ./pretrain.sh, ./midtrain.sh, ./sft.sh, or ./rl.sh"
        exit 1
    fi
fi

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

echo ""
echo "=========================================="
echo "NanoChat OpenAI-Compatible API Service"
echo "=========================================="
echo "Model source:    $SOURCE"
echo "Port:            $PORT"
echo "Host:            $HOST"
echo "Temperature:     $TEMPERATURE"
echo "Top-k:           $TOP_K"
echo "Max tokens:      $MAX_TOKENS"
echo "Data type:       $DTYPE"
echo "=========================================="
echo ""

if [ "$RUN_AS_SERVER" = false ]; then
    echo "💡 Tip: Run with --server to start as background service"
    echo ""
fi

echo "Starting web..."
echo "API endpoint will be: http://$HOST:$PORT/v1/chat/completions"
echo ""

# Build the command
CMD="python -m scripts.chat_web"
CMD="$CMD --source $SOURCE"
CMD="$CMD --port $PORT"
CMD="$CMD --host $HOST"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top-k $TOP_K"
CMD="$CMD --max-tokens $MAX_TOKENS"
CMD="$CMD --dtype $DTYPE"

# Run the service
if [ "$RUN_AS_SERVER" = true ]; then
    # Run as background server with logging
    LOG_FILE="chat_web_${PORT}.log"
    echo "Starting server in background..."
    echo "Log file: $LOG_FILE"
    echo "API endpoint: http://$HOST:$PORT/v1/chat/completions"
    echo ""
    echo "To stop the server, run: ./chat_web.sh --stop"
    echo ""
    
    nohup $CMD > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "✓ Server started with PID: $SERVER_PID"
    echo "  View logs: tail -f $LOG_FILE"
else
    # Run in foreground (interactive mode)
    exec $CMD
fi
