import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Host the trained Overcooked model using SGLang")
    parser.add_argument("--model_path", type=str, default="models/merged_model", 
                        help="Path to the model")
    parser.add_argument("--port", type=int, default=30000, 
                        help="Port to run the SGLang server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host address to bind the server to")
    parser.add_argument("--tp_size", type=int, default=1, 
                        help="Tensor parallelism size")
    parser.add_argument("--quantization", type=str, choices=["fp8", "awq", "gptq", None], default=None,
                        help="Quantization method to use (if any)")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32", "auto"], default="auto",
                        help="Data type to use for model weights")
    parser.add_argument("--context_length", type=int, default=2048,
                        help="Maximum context length for the model")
    args = parser.parse_args()

    # Check if SGLang is installed
    try:
        import sglang as sgl
    except ImportError:
        print("SGLang is not installed. Installing it now...")
        os.system("pip install sglang")
        try:
            import sglang as sgl
        except ImportError:
            print("Failed to install SGLang. Please install it manually with 'pip install sglang'")
            return

    print(f"Starting SGLang server for Overcooked model...")
    
    # Prepare the SGLang server command
    launch_cmd = f"python -m sglang.launch_server " \
                f"--model-path {args.model_path} " \
                f"--port {args.port} " \
                f"--host {args.host} " \
                f"--tensor-parallel-size {args.tp_size} " \
                f"--context-length {args.context_length}"
    
    # Add optional parameters if specified
    if args.quantization:
        launch_cmd += f" --quantization {args.quantization}"
    
    if args.dtype != "auto":
        launch_cmd += f" --dtype {args.dtype}"
    
    print(f"Launching SGLang server with command: {launch_cmd}")
    os.system(launch_cmd)

if __name__ == "__main__":
    main() 