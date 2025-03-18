#!/bin/bash

# Configuration
BASE_MODEL="EleutherAI/pythia-1b-deduped"
ADAPTER_PATH="models/minimal/ppo_tldr"
MERGED_MODEL_PATH="models/merged_model"
PORT=30000
HOST="0.0.0.0"
TP_SIZE=1
DTYPE="float16"  # Options: float16, bfloat16, float32, auto

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --adapter_path)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --merged_model_path)
      MERGED_MODEL_PATH="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --tp_size)
      TP_SIZE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --skip_merge)
      SKIP_MERGE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --base_model MODEL         Base model name or path (default: $BASE_MODEL)"
      echo "  --adapter_path PATH        Path to the LoRA adapter (default: $ADAPTER_PATH)"
      echo "  --merged_model_path PATH   Path to save the merged model (default: $MERGED_MODEL_PATH)"
      echo "  --port PORT                Port to run the SGLang server on (default: $PORT)"
      echo "  --host HOST                Host address to bind to (default: $HOST)"
      echo "  --tp_size SIZE             Tensor parallelism size (default: $TP_SIZE)"
      echo "  --dtype TYPE               Data type to use (default: $DTYPE)"
      echo "  --skip_merge               Skip the model merging step"
      echo "  --help                     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Step 1: Merge LoRA weights with base model (unless skipped)
if [ "$SKIP_MERGE" != "true" ]; then
  echo "Step 1: Merging LoRA weights with base model..."
  python merge_lora_model.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$ADAPTER_PATH" \
    --output_path "$MERGED_MODEL_PATH" \
    --half_precision
  
  # Check if the merge was successful
  if [ $? -ne 0 ]; then
    echo "Error: Failed to merge LoRA weights with base model."
    exit 1
  fi
else
  echo "Skipping model merge step..."
fi

# Step 2: Start the SGLang server with the merged model
echo "Step 2: Starting SGLang server with the merged model..."
python serve_overcooked_model.py \
  --model_path "$MERGED_MODEL_PATH" \
  --port "$PORT" \
  --host "$HOST" \
  --tp_size "$TP_SIZE" \
  --dtype "$DTYPE"

# The script will continue running until the server is stopped 