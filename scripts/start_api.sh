#!/bin/bash
# Start an OpenAI-compatible API for the base model (pre-SFT)

# Default values
DEFAULT_MODEL_PATH="./models/Qwen3-4B"
DEFAULT_PORT=8002

# Parse command line arguments
MODEL_PATH_ARG=""
PORT_ARG=""
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path|-m)
            if [ -z "$2" ]; then
                echo "Error: --model_path requires a value" >&2
                exit 1
            fi
            MODEL_PATH_ARG="$2"
            shift 2
            ;;
        --port|-p)
            if [ -z "$2" ]; then
                echo "Error: --port requires a value" >&2
                exit 1
            fi
            PORT_ARG="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Show help message
if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Start an OpenAI-compatible API server for local LLM models"
    echo ""
    echo "Options:"
    echo "  -m, --model_path PATH    Path to the local LLM model (default: $DEFAULT_MODEL_PATH)"
    echo "  -p, --port PORT         API server port (default: $DEFAULT_PORT)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model_path ./models/Qwen3-1.7B"
    echo "  $0 -m ./models/windy/sft_qwen3_1.7b_windy -p 8003"
    echo ""
    echo "Note: Model path can also be set via .env file (MODEL_PATH) or environment variable"
    exit 0
fi

# Get script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Try to find .env file (current dir, project root, or example path)
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE=".env"
elif [ -f "$PROJECT_ROOT/.env" ]; then
    ENV_FILE="$PROJECT_ROOT/.env"
elif [ -f "/data/Forever_Pan/AGI_sources/CastMaster_new/.env" ]; then
    ENV_FILE="/data/Forever_Pan/AGI_sources/CastMaster_new/.env"
fi

# Load MODEL_PATH from .env file if it exists and not overridden by argument
if [ -z "$MODEL_PATH_ARG" ] && [ -n "$ENV_FILE" ]; then
    # Extract MODEL_PATH from .env (handle quoted and unquoted values)
    MODEL_PATH_FROM_ENV=$(grep -E "^MODEL_PATH=" "$ENV_FILE" 2>/dev/null | sed -E 's/^MODEL_PATH=["'\'']?([^"'\'']+)["'\'']?/\1/' | head -1)
    if [ -n "$MODEL_PATH_FROM_ENV" ]; then
        MODEL_PATH_ARG="$MODEL_PATH_FROM_ENV"
    else
        # If MODEL_PATH not found, try to use MODEL from .env
        MODEL_FROM_ENV=$(grep -E "^MODEL=" "$ENV_FILE" 2>/dev/null | sed -E 's/^MODEL=["'\'']?([^"'\'']+)["'\'']?/\1/' | head -1)
        if [ -n "$MODEL_FROM_ENV" ]; then
            MODEL_PATH_ARG="$MODEL_FROM_ENV"
        fi
    fi
fi

# Determine final MODEL_PATH: command line argument > .env file > environment variable > default
if [ -n "$MODEL_PATH_ARG" ]; then
    MODEL_PATH="$MODEL_PATH_ARG"
elif [ -n "$MODEL_PATH" ]; then
    # Use existing environment variable
    :
else
    MODEL_PATH="$DEFAULT_MODEL_PATH"
fi

# Determine final PORT: command line argument > environment variable > default
if [ -n "$PORT_ARG" ]; then
    PORT="$PORT_ARG"
elif [ -z "$PORT" ]; then
    PORT="$DEFAULT_PORT"
fi

# Validate port is a number (do this first to catch errors early)
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: Port must be a number: $PORT" >&2
    exit 1
fi

# Check if port is in valid range
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo "Error: Port must be between 1 and 65535: $PORT" >&2
    exit 1
fi

# Validate model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH" >&2
    exit 1
fi

# Print configuration
echo "Starting API server..."
echo "  Model Path: $MODEL_PATH"
echo "  Port: $PORT"
echo ""

# Start the API server

vllm serve "$MODEL_PATH" \
  --port "$PORT" \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  --max-num-seqs 150

