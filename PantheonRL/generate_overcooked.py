import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def main():
    parser = argparse.ArgumentParser(description="Generate text with a fine-tuned model")
    parser.add_argument("--model_path", type=str, default="models/minimal/ppo_tldr", help="Path to the saved model")
    parser.add_argument("--base_model", type=str, default="EleutherAI/pythia-1b-deduped", help="Base model name or path")
    parser.add_argument("--use_peft", action="store_true", help="Whether the model uses PEFT/LoRA")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(f"Loading model from {args.model_path}")
    try:
        if args.use_peft:
            print("Loading as PEFT/LoRA model")
            # Load the base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Then load the adapter
            model = PeftModel.from_pretrained(
                base_model,
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Load the model directly
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # Set the model to evaluation mode
    model.eval()

    # Test generation with a few different prompts
    test_prompts = [
        "Move to the onion",
        "Pick up the onion",
        "Move to the pot",
        "Put the onion in the pot"
    ]
    
    for prompt in test_prompts:
        print(f"\nGenerating from prompt: '{prompt}'")
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.inference_mode():
            try:
                outputs = model.generate(
                    inputs,
                    max_new_tokens=20,
                    do_sample=False,  # Use greedy decoding for stability
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode and print the result
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated: {result}")
            except Exception as e:
                print(f"Error during generation: {e}")
                print("Trying with different parameters...")
                try:
                    # Try with simpler parameters
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        num_beams=1,
                    )
                    
                    # Decode and print the result
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Generated: {result}")
                except Exception as e2:
                    print(f"Second attempt also failed: {e2}")

if __name__ == "__main__":
    main() 