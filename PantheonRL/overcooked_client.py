import argparse
import json
import requests
import time

def main():
    parser = argparse.ArgumentParser(description="Client for interacting with the Overcooked model via SGLang")
    parser.add_argument("--server_url", type=str, default="http://localhost:30000",
                        help="URL of the SGLang server")
    parser.add_argument("--prompt", type=str, 
                        help="Custom prompt to send to the model")
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

    print(f"Connecting to SGLang server at {args.server_url}")
    
    # Try both API endpoints that SGLang might expose
    endpoints = [
        "/generate",                  # Basic generation endpoint
        "/v1/chat/completions"        # OpenAI-compatible endpoint
    ]
    
    for prompt in test_prompts:
        print(f"\n--- Generating from prompt: '{prompt}' ---")
        
        for endpoint in endpoints:
            try:
                # Prepare the request based on the endpoint type
                if endpoint == "/generate":
                    # Basic generation endpoint
                    payload = {
                        "text": prompt,
                        "sampling_params": {
                            "max_new_tokens": 50,
                            "temperature": 0.7,
                            "top_p": 0.95
                        }
                    }
                else:
                    # OpenAI-compatible endpoint
                    payload = {
                        "model": "default",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 50
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
                    
                    # Extract the generated text based on the endpoint
                    if endpoint == "/generate":
                        generated_text = response_data.get("text", "")
                        print(f"Generated (via {endpoint}): {prompt}{generated_text}")
                    else:
                        # For OpenAI-compatible endpoint
                        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        print(f"Generated (via {endpoint}): {content}")
                    
                    print(f"Time taken: {end_time - start_time:.2f} seconds")
                    
                    # If successful, no need to try the other endpoint
                    break
                else:
                    print(f"Error with endpoint {endpoint}: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                print(f"Request failed for endpoint {endpoint}: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 