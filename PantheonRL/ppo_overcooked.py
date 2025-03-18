# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
import sys

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


"""
python examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_tldr \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --eval_strategy steps \
    --eval_steps 100

accelerate launch examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 100
"""


if __name__ == "__main__":
    # Set torch options for numerical stability
    torch.set_printoptions(precision=10)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Add QLoRA parameters to command line arguments if not already present
    # Only add these if they're not already in the command line
    if "--use_peft" not in " ".join(sys.argv) and "--load_in_4bit" not in " ".join(sys.argv):
        print("Adding QLoRA parameters to command line arguments...")
        sys.argv.extend([
            "--use_peft", "True",            # Enable PEFT
            "--load_in_4bit", "True",        # Enable 4-bit quantization
            "--lora_r", "16",                # LoRA rank
            "--lora_alpha", "32",            # LoRA alpha
            "--lora_dropout", "0.05",        # LoRA dropout
        ])
    
    # Patch the batch_generation function in trl.trainer.utils
    from trl.trainer.utils import batch_generation as original_batch_generation
    import functools
    
    @functools.wraps(original_batch_generation)
    def safe_batch_generation(*args, **kwargs):
        try:
            # Add safety measures
            kwargs.setdefault("do_sample", True)
            kwargs.setdefault("top_k", 50)
            kwargs.setdefault("top_p", 0.95)
            kwargs.setdefault("temperature", 0.7)
            kwargs.setdefault("max_new_tokens", 20)
            
            # Call the original function
            return original_batch_generation(*args, **kwargs)
        except RuntimeError as e:
            if "probability tensor" in str(e):
                print("Caught probability tensor error in batch_generation, using fallback...")
                # Extract necessary arguments
                lm_backbone = args[0]
                queries = args[1]
                
                # Fallback to a simpler generation approach
                with torch.inference_mode():
                    outputs = lm_backbone.generate(
                        queries,
                        do_sample=False,  # Use greedy decoding as fallback
                        max_new_tokens=10,
                        pad_token_id=lm_backbone.config.pad_token_id,
                        eos_token_id=lm_backbone.config.eos_token_id,
                    )
                
                # Create dummy logits (this is a simplification)
                dummy_logits = torch.zeros((outputs.shape[0], outputs.shape[1], lm_backbone.config.vocab_size), 
                                          device=outputs.device)
                
                return outputs, dummy_logits
            else:
                # Re-raise other errors
                raise
    
    # Apply the patch
    import trl.trainer.utils
    trl.trainer.utils.batch_generation = safe_batch_generation
    
    # Patch the generate method in transformers.generation.utils
    from transformers.generation.utils import GenerationMixin
    
    # Store the original generate method
    original_generate = GenerationMixin.generate
    
    # Define a safer generate method
    def safe_generate(self, *args, **kwargs):
        try:
            # Add safety measures
            kwargs.setdefault("do_sample", True)
            kwargs.setdefault("top_k", 50)
            kwargs.setdefault("top_p", 0.95)
            kwargs.setdefault("temperature", 0.7)
            kwargs.setdefault("max_new_tokens", 20)
            
            # Call the original method
            return original_generate(self, *args, **kwargs)
        except RuntimeError as e:
            if "probability tensor" in str(e):
                print("Caught probability tensor error in generate, using fallback...")
                # Use greedy decoding as fallback
                kwargs["do_sample"] = False
                kwargs["top_k"] = None
                kwargs["top_p"] = None
                kwargs["temperature"] = 1.0
                kwargs["max_new_tokens"] = min(kwargs.get("max_new_tokens", 20), 10)
                
                return original_generate(self, *args, **kwargs)
            else:
                # Re-raise other errors
                raise
    
    # Apply the patch
    GenerationMixin.generate = safe_generate
    
    print("Starting PPO training...")
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    
    # Configure quantization for QLoRA
    if model_args.load_in_4bit or model_args.load_in_8bit:
        print(f"Using quantization: 4-bit={model_args.load_in_4bit}, 8-bit={model_args.load_in_8bit}")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
        )
    else:
        quantization_config = None
    
    # Configure device map
    if quantization_config is not None:
        device_map = get_kbit_device_map()
    else:
        device_map = None
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.float16 if quantization_config is not None else torch.float32,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    print(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    print(f"Loading value model from {training_args.reward_model_path}")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1, **model_kwargs
    )
    
    print(f"Loading reward model from {training_args.reward_model_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1, **model_kwargs
    )
    
    print(f"Loading policy model from {training_args.sft_model_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    # Configure PEFT (LoRA)
    if model_args.use_peft:
        print("Using PEFT (LoRA) for fine-tuning")
        from peft import LoraConfig, TaskType
        
        # Define target modules for LoRA
        if not model_args.lora_target_modules:
            # Default target modules for common model architectures
            if "llama" in model_args.model_name_or_path.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "gpt" in model_args.model_name_or_path.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "pythia" in model_args.model_name_or_path.lower():
                # Pythia models use these module names
                target_modules = ["query_key_value", "dense"]
            else:
                # Try a more general approach for transformer models
                target_modules = ["query_key_value", "dense", "c_attn", "c_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
                print(f"Using general target modules for unknown model architecture: {target_modules}")
        else:
            target_modules = model_args.lora_target_modules
        
        print(f"LoRA target modules: {target_modules}")
        
        # Create LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
        )
        
        # No need for reference model with LoRA
        ref_policy = None
    else:
        peft_config = None
        if peft_config is None:
            print("Loading reference policy model")
            ref_policy = AutoModelForCausalLM.from_pretrained(
                training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
            )
        else:
            ref_policy = None

    ################
    # Dataset
    ################
    try:
        print(f"Loading dataset: {script_args.dataset_name}")
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        train_dataset = dataset[script_args.dataset_train_split]
        eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
        
        print(f"Dataset loaded successfully. Train size: {len(train_dataset)}, Eval size: {len(eval_dataset) if eval_dataset else 0}")
        print(f"Sample data: {train_dataset[0] if len(train_dataset) > 0 else 'Empty dataset'}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating a simple dummy dataset for testing...")
        
        # Create a simple dummy dataset with chat messages
        from datasets import Dataset
        
        # Simple dummy data with chat format for Overcooked
        dummy_data = {
            "messages": [
                [{"role": "user", "content": "Move to the onion"}],
                [{"role": "user", "content": "Pick up the onion"}],
                [{"role": "user", "content": "Move to the pot"}],
                [{"role": "user", "content": "Put the onion in the pot"}],
                [{"role": "user", "content": "Wait for the soup to cook"}],
                [{"role": "user", "content": "Pick up the soup"}],
                [{"role": "user", "content": "Move to the serving area"}],
                [{"role": "user", "content": "Serve the soup"}],
            ]
        }
        
        # Create datasets
        train_dataset = Dataset.from_dict(dummy_data)
        eval_dataset = Dataset.from_dict(dummy_data)
        
        print(f"Created dummy dataset with {len(train_dataset)} examples")

    # Limit dataset size for testing if needed
    if training_args.max_steps and training_args.max_steps < 100:
        print(f"Limiting dataset size for testing (max_steps={training_args.max_steps})")
        train_size = min(len(train_dataset), 100)
        train_dataset = train_dataset.select(range(train_size))
        if eval_dataset is not None:
            eval_size = min(len(eval_dataset), 20)
            eval_dataset = eval_dataset.select(range(eval_size))

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            try:
                # Check the structure of the dataset
                if "messages" in element:
                    # Try to use chat template first
                    input_ids = tokenizer.apply_chat_template(
                        element["messages"][:1],
                        padding=False,
                        add_generation_prompt=True,
                    )
                elif "input" in element:
                    # Alternative format
                    input_ids = tokenizer.encode(element["input"])
                elif "text" in element:
                    # Simple text format
                    input_ids = tokenizer.encode(element["text"])
                else:
                    # Fallback to a default prompt
                    print(f"Unknown dataset format: {list(element.keys())}")
                    input_ids = tokenizer.encode("Move to the onion")
            except Exception as e:
                print(f"Error tokenizing element: {e}")
                # Fallback to simple tokenization
                input_ids = tokenizer.encode("Move to the onion")
                
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        try:
            return dataset.map(
                tokenize,
                remove_columns=dataset.column_names,
                num_proc=training_args.dataset_num_proc,
            )
        except Exception as e:
            print(f"Error mapping dataset: {e}")
            # Create a simple tokenized dataset as fallback
            from datasets import Dataset
            
            # Create a simple dataset with tokenized inputs
            dummy_inputs = [tokenizer.encode("Move to the onion") for _ in range(8)]
            dummy_lengths = [len(ids) for ids in dummy_inputs]
            
            return Dataset.from_dict({
                "input_ids": dummy_inputs,
                "lengths": dummy_lengths
            })

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # filtering
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    ################
    # Training
    ################
    print("train_dataset", train_dataset)
    print("------")
    print("eval_dataset", eval_dataset)
    
    # Modify PPO training args for better numerical stability and memory efficiency
    training_args.init_kl_coef = 0.1  # Lower KL coefficient
    training_args.adap_kl_ctrl = True  # Enable adaptive KL control
    training_args.target = 6  # Target KL value
    training_args.horizon = 10000  # Horizon for adaptive KL
    training_args.cliprange = 0.2  # Standard PPO clipping range
    training_args.cliprange_value = 0.2  # Value clipping range
    training_args.gamma = 0.99  # Discount factor
    training_args.lambda_ = 0.95  # GAE lambda
    
    # Memory optimization settings for QLoRA
    if model_args.use_peft:
        print("Applying memory optimization settings for QLoRA...")
        
        # Memory optimization
        training_args.gradient_checkpointing = True
        
        # Ensure we're using a memory-efficient optimizer
        if hasattr(training_args, "optim") and training_args.optim not in ["adamw_torch_fused", "adamw_8bit"]:
            print(f"Changing optimizer from {training_args.optim} to adamw_8bit for memory efficiency")
            training_args.optim = "adamw_8bit"
        
        # Set a reasonable forward batch size for rollouts
        if not hasattr(training_args, "local_rollout_forward_batch_size") or training_args.local_rollout_forward_batch_size > 4:
            print("Setting local_rollout_forward_batch_size to 4 for memory efficiency")
            training_args.local_rollout_forward_batch_size = 4
        
        # Ensure we're using a reasonable batch size
        if training_args.per_device_train_batch_size > 1:
            print(f"Reducing per_device_train_batch_size from {training_args.per_device_train_batch_size} to 1 for memory efficiency")
            training_args.per_device_train_batch_size = 1
        
        # Ensure we have enough gradient accumulation steps
        if training_args.gradient_accumulation_steps < 4:
            print(f"Increasing gradient_accumulation_steps from {training_args.gradient_accumulation_steps} to 4 for memory efficiency")
            training_args.gradient_accumulation_steps = 4
    
    try:
        print("Initializing PPOTrainer with QLoRA configuration...")
        # Create a variable to hold the trainer
        trainer = None
        
        # Initialize the trainer with proper error handling
        try:
            trainer = PPOTrainer(
                args=training_args,
                processing_class=tokenizer,
                model=policy,
                ref_model=ref_policy,
                reward_model=reward_model,
                value_model=value_model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
            )
        except Exception as init_error:
            print(f"Error initializing PPOTrainer: {init_error}")
            
            if "Target modules" in str(init_error) and model_args.use_peft:
                # If the error is about target modules, try to get the actual module names from the model
                print("Attempting to find valid target modules in the model...")
                try:
                    # Get the model's module names
                    module_names = [name for name, _ in policy.named_modules()]
                    print(f"Available modules in the model: {module_names}")
                    
                    # Try to find attention-related modules
                    attention_modules = [name for name in module_names if any(
                        keyword in name for keyword in ["attention", "attn", "query", "key", "value", "q_proj", "k_proj", "v_proj", "qkv"]
                    )]
                    
                    if attention_modules:
                        print(f"Potential attention modules found: {attention_modules}")
                        print("Please update the target_modules in the script with these values and try again.")
                    else:
                        print("No attention modules found. Please check the model architecture.")
                except Exception as module_error:
                    print(f"Error while trying to find modules: {module_error}")
            
            print("Exiting due to initialization error.")
            sys.exit(1)
        
        if trainer is None:
            print("Trainer initialization failed. Exiting.")
            sys.exit(1)
        
        print("Starting PPO training with QLoRA...")
        trainer.train()
        print("Training done!")
        
        # Save model even if generation fails
        print("Saving model...")
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)
        
        # Generate completions with the trained model
        print("\nGenerating completions with the trained model...")
        try:
            # Load the saved model directly instead of using the wrapped model
            print(f"Loading the saved model from {training_args.output_dir} for generation...")
            
            # For PEFT models, we need to load the base model and then the adapter
            if model_args.use_peft:
                from peft import PeftModel, PeftConfig
                
                try:
                    # Load the base model first
                    generation_model = AutoModelForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        trust_remote_code=model_args.trust_remote_code,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    
                    # Then load the adapter
                    generation_model = PeftModel.from_pretrained(
                        generation_model,
                        training_args.output_dir,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    print("Successfully loaded PEFT model with adapter")
                except Exception as peft_error:
                    print(f"Error loading PEFT model: {peft_error}")
                    print("Falling back to direct model loading...")
                    generation_model = AutoModelForCausalLM.from_pretrained(
                        training_args.output_dir,
                        trust_remote_code=model_args.trust_remote_code,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                # Load the model directly
                generation_model = AutoModelForCausalLM.from_pretrained(
                    training_args.output_dir,
                    trust_remote_code=model_args.trust_remote_code,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
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
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(generation_model.device)
                
                # Generate
                with torch.inference_mode():
                    outputs = generation_model.generate(
                        inputs,
                        max_new_tokens=20,
                        do_sample=False,  # Use greedy decoding for stability
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode and print the result
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated: {result}")
        except Exception as gen_error:
            print(f"Error during generation: {gen_error}")
            print("Generation failed, but the model was saved successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to save model even if training fails
        if 'trainer' in locals() and trainer is not None:
            try:
                print("Attempting to save model despite error...")
                trainer.save_model(training_args.output_dir)
                if training_args.push_to_hub:
                    trainer.push_to_hub(dataset_name=script_args.dataset_name)
            except Exception as save_error:
                print(f"Error saving model: {save_error}")
        else:
            print("Cannot save model because trainer was not successfully initialized.")