import argparse
import json
import requests
import time

def main():
    parser = argparse.ArgumentParser(description="Client for interacting with the Overcooked model via vLLM")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000",
                        help="URL of the vLLM server")
    parser.add_argument("--prompt", type=str, 
                        help="Custom prompt to send to the model")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum number of tokens to generate")
    args = parser.parse_args()

    # Define test prompts for Overcooked
    test_prompts = [
        "Move to the onion",
        "Pick up the onion",
        "Move to the pot",
        "Put the onion in the pot",
        "Wait for the soup to cook",
        "Pick up the soup",
        "Move to the serving area",
        "Serve the soup"
    ]

    # If a custom prompt is provided, use it instead
    if args.prompt:
        test_prompts = [args.prompt]

    print(f"Connecting to vLLM server at {args.server_url}")
    
    # vLLM uses the OpenAI-compatible API
    endpoint = "/v1/completions"
    
    for prompt in test_prompts:
        print(f"\n--- Generating from prompt: '{prompt}' ---")
        
        try:
            # Prepare the request
            payload = {
                "model": "overcooked",  # This is just a placeholder, vLLM ignores it
                "prompt": prompt,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stream": False
            }
            
            # Send the request
            start_time = time.time()
            response = requests.post(
                f"{args.server_url}{endpoint}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )
            end_time = time.time()
            
            # Process the response
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract the generated text
                generated_text = response_data.get("choices", [{}])[0].get("text", "")
                print(f"Generated: {prompt}{generated_text}")
                print(f"Time taken: {end_time - start_time:.2f} seconds")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        
        print("-" * 50)

    # Also try the chat completions API
    chat_endpoint = "/v1/chat/completions"
    
    for prompt in test_prompts:
        print(f"\n--- Generating chat response for: '{prompt}' ---")
        
        try:
            # Prepare the request
            payload = {
                "model": "overcooked",  # This is just a placeholder, vLLM ignores it
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stream": False
            }
            
            # Send the request
            start_time = time.time()
            response = requests.post(
                f"{args.server_url}{chat_endpoint}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )
            end_time = time.time()
            
            # Process the response
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract the generated text
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"Generated: {content}")
                print(f"Time taken: {end_time - start_time:.2f} seconds")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 