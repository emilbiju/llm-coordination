# Hosting Overcooked Model with SGLang

This README explains how to host the trained Overcooked model using SGLang and interact with it.

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- PEFT (if using LoRA/QLoRA models)
- SGLang (will be installed automatically by the scripts if not present)

## Installation

```bash
pip install torch transformers peft requests
pip install sglang
```

## Scripts Overview

This repository contains four scripts for hosting and interacting with the Overcooked model:

1. `merge_lora_model.py`: Merges LoRA weights with the base model (required for PEFT/LoRA models)
2. `serve_overcooked_model.py`: Hosts the model using SGLang server
3. `overcooked_client.py`: Simple client to interact with the hosted model
4. `overcooked_sglang_frontend.py`: Advanced client using SGLang's frontend language for structured generation

## Hosting the Model

### Step 1: Merge LoRA Weights (for PEFT/LoRA models only)

If your model was trained with PEFT/LoRA (like the QLoRA approach used in training), you need to merge the weights first:

```bash
# Merge LoRA weights with base model
python merge_lora_model.py --base_model EleutherAI/pythia-1b-deduped --adapter_path models/minimal/ppo_tldr --output_path models/merged_model --half_precision
```

This will create a new model with the LoRA weights merged into the base model, which can then be served with SGLang.

### Step 2: Serve the Model

To host the model, run:

```bash
# For a merged model or regular model
python serve_overcooked_model.py --model_path models/merged_model
```

Additional options:
- `--port`: Port to run the server on (default: 30000)
- `--host`: Host address to bind to (default: 0.0.0.0)
- `--tp_size`: Tensor parallelism size for multi-GPU deployment (default: 1)
- `--quantization`: Quantization method to use (fp8, awq, gptq)
- `--dtype`: Data type to use (float16, bfloat16, float32, auto)
- `--context_length`: Maximum context length (default: 2048)

## Interacting with the Model

### Using the Simple Client

Once the model is hosted, you can interact with it using the simple client:

```bash
python overcooked_client.py

# To use a custom prompt
python overcooked_client.py --prompt "Move to the tomato"

# If the server is running on a different host/port
python overcooked_client.py --server_url http://different-host:12345
```

### Using the SGLang Frontend

For more advanced structured generation, you can use the SGLang frontend:

```bash
python overcooked_sglang_frontend.py

# To use a custom prompt
python overcooked_sglang_frontend.py --prompt "Move to the tomato"

# If the server is running on a different host/port
python overcooked_sglang_frontend.py --server_url http://different-host:12345
```

## Example Workflow

### Using Individual Scripts

1. Merge the LoRA weights with the base model:
   ```bash
   python merge_lora_model.py --base_model EleutherAI/pythia-1b-deduped --adapter_path models/minimal/ppo_tldr --output_path models/merged_model --half_precision
   ```

2. Start the server with the merged model:
   ```bash
   python serve_overcooked_model.py --model_path models/merged_model
   ```

3. In another terminal, interact with the model:
   ```bash
   python overcooked_client.py
   ```

4. Try structured generation:
   ```bash
   python overcooked_sglang_frontend.py
   ```

### Using the Automated Shell Script

For convenience, you can use the provided shell script to automate the process:

```bash
# Run with default settings
./run_overcooked_server.sh

# Skip the merge step if you've already merged the model
./run_overcooked_server.sh --skip_merge

# Customize the parameters
./run_overcooked_server.sh --base_model EleutherAI/pythia-1b-deduped --adapter_path models/minimal/ppo_tldr --port 8080
```

Run `./run_overcooked_server.sh --help` to see all available options.

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try using half precision (`--half_precision` when merging) or quantization options (`--quantization fp8` when serving).
- If the server fails to start, check if the model path is correct and if you've specified the correct options for your model type.
- If the client can't connect to the server, ensure the server is running and the host/port settings are correct.
- If you get errors about missing parameters when launching the server, check the SGLang documentation for the correct parameter names, as they may change with different versions.

## Advanced Usage

SGLang supports various advanced features:

- Quantization: Use `--quantization fp8` or other quantization options to reduce memory usage
- Multi-GPU deployment: Increase `--tp_size` for tensor parallelism
- Custom chat templates: Use `--chat-template` to specify a custom template

For more details, refer to the [SGLang documentation](https://docs.sglang.ai/). 