import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Host the trained Overcooked model using vLLM")
    parser.add_argument("--model_path", type=str, default="models/merged_model", 
                        help="Path to the model")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port to run the vLLM server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host address to bind the server to")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                        help="Tensor parallelism size")
    parser.add_argument("--quantization", type=str, choices=["awq", "gptq", "squeezellm", None], default=None,
                        help="Quantization method to use (if any)")
    parser.add_argument("--dtype", type=str, choices=["half", "float16", "bfloat16", "float", "auto"], default="auto",
                        help="Data type to use for model weights")
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use (0.0 to 1.0)")
    args = parser.parse_args()

    # Check if vLLM is installed
    try:
        import vllm
    except ImportError:
        print("vLLM is not installed. Installing it now...")
        os.system("pip install vllm")
        try:
            import vllm
        except ImportError:
            print("Failed to install vLLM. Please install it manually with 'pip install vllm'")
            return

    print(f"Starting vLLM server for Overcooked model...")
    
    # Prepare the vLLM server command
    launch_cmd = f"python -m vllm.entrypoints.openai.api_server " \
                f"--model {args.model_path} " \
                f"--port {args.port} " \
                f"--host {args.host} " \
                f"--tensor-parallel-size {args.tensor_parallel_size} " \
                f"--max-model-len {args.max_model_len} " \
                f"--gpu-memory-utilization {args.gpu_memory_utilization}"
    
    # Add optional parameters if specified
    if args.quantization:
        launch_cmd += f" --quantization {args.quantization}"
    
    if args.dtype != "auto":
        launch_cmd += f" --dtype {args.dtype}"
    
    print(f"Launching vLLM server with command: {launch_cmd}")
    os.system(launch_cmd)

if __name__ == "__main__":
    main() 