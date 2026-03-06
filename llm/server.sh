#!/bin/bash
#
# vLLM Server Manager for NVIDIA DGX Spark
# Manages vLLM container as a background service
#

set -e

# Default values
CONTAINER_NAME="vllm-server"
IMAGE_NAME="vllm-custom:latest"
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
MAX_LEN="32768"
PORT="8000"

# Function to check if container is running
is_running() {
    docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if container exists (but may be stopped)
container_exists() {
    docker ps -a --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to clear system memory caches
clear_memory() {
    echo "Clearing system memory caches..."
    if sudo -n sysctl -w vm.drop_caches=3 > /dev/null 2>&1; then
        echo "✓ Memory caches cleared"
    else
        echo "Note: Could not clear memory caches (may require sudo password)"
        echo "      Run manually: sudo sysctl -w vm.drop_caches=3"
    fi
    echo ""
}

# Function to start the server
start_server() {
    if is_running; then
        echo "✓ vLLM server is already running"
        echo "  Container: ${CONTAINER_NAME}"
        echo "  API: http://localhost:${PORT}"
        return 0
    fi

    if container_exists; then
        echo "Starting existing container..."
        docker start ${CONTAINER_NAME}
        echo "✓ vLLM server started"
    else
        # Clear memory before starting new server for optimal performance
        clear_memory
        
        echo "Starting new vLLM server..."
        echo "  Model: ${MODEL}"
        echo "  Max length: ${MAX_LEN}"
        echo "  Port: ${PORT}"
        echo ""
        
        docker run -d --name ${CONTAINER_NAME} \
            --gpus all -p ${PORT}:8000 \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            -v ~/.cache/vllm:/root/.cache/vllm \
	        --restart unless-stopped \
            ${IMAGE_NAME} \
            vllm serve "${MODEL}" --max-model-len "${MAX_LEN}" --chat-template-content-format "openai"
        
        echo "✓ vLLM server started"
    fi
    
    echo ""
    echo "API endpoint: http://localhost:${PORT}/v1"
    echo ""
    echo "Useful commands:"
    echo "  View logs:   docker logs -f ${CONTAINER_NAME}"
    echo "  Stop server: docker stop ${CONTAINER_NAME}"
    echo "  Restart:     docker restart ${CONTAINER_NAME}"
    echo "  Remove:      docker rm ${CONTAINER_NAME}"
}

# Function to stop the server
stop_server() {
    if is_running; then
        echo "Stopping vLLM server..."
        docker stop ${CONTAINER_NAME}
        echo "✓ Server stopped"
    else
        echo "Server is not running"
    fi
}

# Function to show server status
show_status() {
    echo "vLLM Server Status"
    echo "=================="
    echo ""
    
    if is_running; then
        echo "Status: RUNNING ✓"
        echo ""
        docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        echo "API endpoint: http://localhost:${PORT}/v1"
        echo ""
        echo "View logs: docker logs -f ${CONTAINER_NAME}"
    elif container_exists; then
        echo "Status: STOPPED"
        echo ""
        docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}"
        echo ""
        echo "Start with: $0 start"
    else
        echo "Status: NOT CREATED"
        echo ""
        echo "Create and start with: $0 start --model <model> --max-len <length> --port <port>"
    fi
}

# Function to show logs
show_logs() {
    if container_exists; then
        echo "Showing logs for ${CONTAINER_NAME} (Ctrl+C to exit)..."
        echo ""
        docker logs -f ${CONTAINER_NAME}
    else
        echo "Container does not exist. Start the server first with: $0 start"
        exit 1
    fi
}

# Function to restart the server
restart_server() {
    if container_exists; then
        echo "Restarting vLLM server..."
        docker restart ${CONTAINER_NAME}
        echo "✓ Server restarted"
        echo ""
        echo "View logs: docker logs -f ${CONTAINER_NAME}"
    else
        echo "Container does not exist. Start the server first with: $0 start"
        exit 1
    fi
}

# Function to remove the server
remove_server() {
    if is_running; then
        echo "Stopping server..."
        docker stop ${CONTAINER_NAME}
    fi
    
    if container_exists; then
        echo "Removing container..."
        docker rm ${CONTAINER_NAME}
        echo "✓ Server removed"
    else
        echo "Container does not exist"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
vLLM Server Manager for DGX Spark

Usage: $0 <command> [options]

Commands:
    start                           Start the vLLM server
    stop                            Stop the server
    restart                         Restart the server
    status                          Show server status
    logs                            Show server logs (follow mode)
    remove                          Stop and remove the server
    help                            Show this help message

Start Options:
    -m, --model <model>             Model name (default: ${MODEL})
    -l, --max-len <length>          Maximum sequence length (default: ${MAX_LEN})
    -p, --port <port>               Host port to expose (default: ${PORT})
    -n, --name <name>               Container name (default: ${CONTAINER_NAME})

Examples:
    # Start with default model
    $0 start

    # Start with custom model
    $0 start --model nvidia/Llama-3.1-8B-Instruct-FP8 --max-len 8192 --port 8888

    # Start with short options
    $0 start -m mistralai/Mistral-7B-Instruct-v0.3 -l 8192 -p 9000

    # Check status
    $0 status

    # View logs
    $0 logs

    # Stop server
    $0 stop

    # Remove server completely
    $0 remove

Once started, the server runs in the background and will be accessible at:
    http://localhost:<port>/v1

EOF
}

# Main command handling
COMMAND="${1:-help}"
shift 2>/dev/null || true

# Parse options for start command
if [ "$COMMAND" = "start" ]; then
    while [ $# -gt 0 ]; do
        case "$1" in
            -m|--model)
                MODEL="$2"
                shift 2
                ;;
            -l|--max-len)
                MAX_LEN="$2"
                shift 2
                ;;
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            -n|--name)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done
fi

case "$COMMAND" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    remove)
        remove_server
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac
