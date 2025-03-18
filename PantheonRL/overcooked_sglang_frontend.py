import argparse
import os
import sys
import json

def main():
    parser = argparse.ArgumentParser(description="Use SGLang frontend with the Overcooked model")
    parser.add_argument("--server_url", type=str, default="http://localhost:30000",
                        help="URL of the SGLang server")
    parser.add_argument("--prompt", type=str, 
                        help="Custom prompt to send to the model")
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

    # Set up the SGLang runtime
    sgl.set_default_backend(sgl.Runtime(args.server_url))

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

    # Define a structured generation function using SGLang
    @sgl.function
    def overcooked_action(state):
        """Generate an action for the Overcooked game based on the current state."""
        # Get the user instruction
        user_instruction = state.user_instruction
        
        # Generate the action
        state.action = sgl.gen(f"{user_instruction}", max_tokens=50, temperature=0.7)
        
        # Parse the action into a structured format
        state.parsed_action = sgl.select(
            ["MOVE", "PICK", "PUT", "WAIT", "SERVE"],
            name="action_type"
        )
        
        # Generate a description of what the action does
        state.description = sgl.gen(
            f"Description of what happens when I {state.parsed_action}: ",
            max_tokens=50,
            temperature=0.7
        )
        
        return {
            "instruction": user_instruction,
            "action": state.action,
            "action_type": state.parsed_action,
            "description": state.description
        }

    # Run the function for each prompt
    print("Using SGLang frontend for structured generation with the Overcooked model")
    
    for prompt in test_prompts:
        print(f"\n--- Processing instruction: '{prompt}' ---")
        
        try:
            # Run the structured generation
            result = overcooked_action.run(user_instruction=prompt)
            
            # Print the result in a structured format
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error processing instruction: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 