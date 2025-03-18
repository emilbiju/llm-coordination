import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument("--base_model", type=str, default="EleutherAI/pythia-1b-deduped", 
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default="models/minimal/ppo_tldr", 
                        help="Path to the LoRA adapter")
    parser.add_argument("--output_path", type=str, default="models/merged_model", 
                        help="Path to save the merged model")
    parser.add_argument("--half_precision", action="store_true", 
                        help="Load and save model in half precision (float16)")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model}")
    
    # Determine the dtype to use
    dtype = torch.float16 if args.half_precision else torch.float32
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto"  # Let the model decide how to map to available devices
    )
    
    print(f"Loading LoRA adapter from {args.adapter_path}")
    
    # Load the LoRA adapter onto the base model
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    print("Merging LoRA weights with base model...")
    
    # Merge the LoRA weights with the base model
    merged_model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Saving merged model to {args.output_path}")
    
    # Save the merged model
    merged_model.save_pretrained(args.output_path)
    
    # Save the tokenizer as well
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_path)
    
    print("Done! The merged model can now be served with SGLang.")
    print(f"To serve the model, run: python serve_overcooked_model.py --model_path {args.output_path}")

if __name__ == "__main__":
    main() 