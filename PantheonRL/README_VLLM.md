# Hosting Overcooked Model with vLLM

This README explains how to host the trained Overcooked model using vLLM and interact with it.

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- PEFT (if using LoRA/QLoRA models)
- vLLM (will be installed automatically by the scripts if not present)

## Installation

```bash
pip install torch transformers peft requests
pip install vllm
```

## Scripts Overview

This repository contains four scripts for hosting and interacting with the Overcooked model:

1. `merge_lora_model.py`: Merges LoRA weights with the base model (required for PEFT/LoRA models)
2. `serve_overcooked_vllm.py`: Hosts the model using vLLM server
3. `overcooked_vllm_client.py`: Client to interact with the hosted model
4. `run_overcooked_vllm.sh`: Shell script to automate the process of merging and serving

## Hosting the Model

### Step 1: Merge LoRA Weights (for PEFT/LoRA models only)

If your model was trained with PEFT/LoRA (like the QLoRA approach used in training), you need to merge the weights first:

```bash
# Merge LoRA weights with base model
python merge_lora_model.py --base_model EleutherAI/pythia-1b-deduped --adapter_path models/minimal/ppo_tldr --output_path models/merged_model --half_precision
```

This will create a new model with the LoRA weights merged into the base model, which can then be served with vLLM.

### Step 2: Serve the Model

To host the model, run:

```bash
# For a merged model or regular model
python serve_overcooked_vllm.py --model_path models/merged_model
```

Additional options:
- `--port`: Port to run the server on (default: 8000)
- `--host`: Host address to bind to (default: 0.0.0.0)
- `--tensor_parallel_size`: Tensor parallelism size for multi-GPU deployment (default: 1)
- `--quantization`: Quantization method to use (awq, gptq, squeezellm)
- `--dtype`: Data type to use (half, float16, bfloat16, float, auto)
- `--max_model_len`: Maximum sequence length (default: 2048)
- `--gpu_memory_utilization`: Fraction of GPU memory to use (default: 0.9)

## Interacting with the Model

Once the model is hosted, you can interact with it using the client:

```bash
python overcooked_vllm_client.py

# To use a custom prompt
python overcooked_vllm_client.py --prompt "Move to the tomato"

# If the server is running on a different host/port
python overcooked_vllm_client.py --server_url http://different-host:8000

# To customize generation parameters
python overcooked_vllm_client.py --temperature 0.5 --max_tokens 100
```

## Example Workflow

### Using Individual Scripts

1. Merge the LoRA weights with the base model:
   ```bash
   python merge_lora_model.py --base_model EleutherAI/pythia-1b-deduped --adapter_path models/minimal/ppo_tldr --output_path models/merged_model --half_precision
   ```

2. Start the server with the merged model:
   ```bash
   python serve_overcooked_vllm.py --model_path models/merged_model
   ```

3. In another terminal, interact with the model:
   ```bash
   python overcooked_vllm_client.py
   ```

### Using the Automated Shell Script

For convenience, you can use the provided shell script to automate the process:

```bash
# Run with default settings
./run_overcooked_vllm.sh

# Skip the merge step if you've already merged the model
./run_overcooked_vllm.sh --skip_merge

# Customize the parameters
./run_overcooked_vllm.sh --base_model EleutherAI/pythia-1b-deduped --adapter_path models/minimal/ppo_tldr --port 8080
```

Run `./run_overcooked_vllm.sh --help` to see all available options.

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try using half precision (`--half_precision` when merging) or reducing `--gpu_memory_utilization` (e.g., to 0.7).
- If the server fails to start, check if the model path is correct and if you've specified the correct options for your model type.
- If the client can't connect to the server, ensure the server is running and the host/port settings are correct.
- If you get errors about missing parameters when launching the server, check the vLLM documentation for the correct parameter names, as they may change with different versions.

## Why vLLM?

vLLM is an excellent choice for serving LLMs because:

1. **High Performance**: vLLM implements PagedAttention for efficient memory management, allowing for higher throughput and lower latency.
2. **OpenAI-compatible API**: vLLM provides an API that's compatible with OpenAI's API, making it easy to integrate with existing applications.
3. **Continuous Batching**: vLLM supports continuous batching, which allows it to efficiently process requests as they arrive.
4. **Quantization Support**: vLLM supports various quantization methods (AWQ, GPTQ, SqueezeLLM) to reduce memory usage and increase inference speed.
5. **Tensor Parallelism**: vLLM supports tensor parallelism for multi-GPU inference, allowing you to serve larger models.

## Advanced Usage

vLLM supports various advanced features:

- Quantization: Use `--quantization awq` or other quantization options to reduce memory usage
- Multi-GPU deployment: Increase `--tensor_parallel_size` for tensor parallelism
- Custom prompt templates: vLLM supports various prompt templates through its API

For more details, refer to the [vLLM documentation](https://vllm.readthedocs.io/). 