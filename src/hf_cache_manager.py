import os
import torch
from typing import Dict, Any, Optional, Tuple, List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
import time
import hashlib

class HFCacheManager:
    """
    Manages Key-Value Cache for HuggingFace transformers models.
    Implements Cache-Augmented Generation (CAG) approach.
    """
    
    def __init__(self, cache_dir: str = "./cache/hf_cache"):
        """
        Initialize the HuggingFace Cache Manager.
        
        Args:
            cache_dir: Directory to store transformer KV caches
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def preprocess_knowledge(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        knowledge: str,
        system_prompt: Optional[str] = None
    ) -> Tuple[DynamicCache, int]:
        """
        Preprocess knowledge into KV cache for later use.
        
        Args:
            model: HuggingFace model
            tokenizer: Associated tokenizer
            knowledge: Knowledge text to preprocess
            system_prompt: Optional system prompt
            
        Returns:
            Tuple of (kv_cache, cache_length)
        """
        # Determine the device being used
        embed_device = next(model.parameters()).device
        
        # Create the full prompt with knowledge
        if system_prompt:
            # Format knowledge with system instructions (adapt based on model type)
            if "llama" in model.config.name_or_path.lower():
                # Llama-style formatting
                full_prompt = f"<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\nContext information is below.\n------------------------------------------------\n{knowledge}\n------------------------------------------------\nQuestion:"
            else:
                # Default formatting for other models
                full_prompt = f"System: {system_prompt}\n\nKnowledge:\n{knowledge}\n\nQuestion:"
        else:
            # Simple knowledge format without system instructions
            full_prompt = f"Knowledge:\n{knowledge}\n\nQuestion:"
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(embed_device)
        
        # Initialize KV cache
        past_key_values = DynamicCache()
        
        # Run the model to populate the KV cache
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
        
        # Get the length of the KV cache for reset after generation
        kv_len = outputs.past_key_values.key_cache[0].shape[-2]
        
        return outputs.past_key_values, kv_len
    
    def clean_up_cache(self, kv_cache: DynamicCache, origin_len: int) -> DynamicCache:
        """
        Reset the KV cache to its original length to avoid contamination between queries.
        
        Args:
            kv_cache: The KV cache to reset
            origin_len: The original length to truncate to
            
        Returns:
            The reset KV cache
        """
        for i in range(len(kv_cache.key_cache)):
            kv_cache.key_cache[i] = kv_cache.key_cache[i][:, :, :origin_len, :]
            kv_cache.value_cache[i] = kv_cache.value_cache[i][:, :, :origin_len, :]
        return kv_cache
    
    def generate_with_knowledge(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        query: str,
        kv_cache: DynamicCache,
        kv_cache_len: int,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate text using the preloaded knowledge in KV cache.
        
        Args:
            model: HuggingFace model
            tokenizer: Associated tokenizer
            query: User query
            kv_cache: Preloaded knowledge KV cache
            kv_cache_len: Original length of the KV cache
            max_new_tokens: Maximum number of tokens to generate
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with generation results
        """
        start_time = time.time()
        
        # Reset the KV cache to its original state
        kv_cache = self.clean_up_cache(kv_cache, kv_cache_len)
        
        # Determine the device being used
        embed_device = next(model.parameters()).device
        
        # Tokenize user query
        input_ids = tokenizer.encode(query, return_tensors="pt").to(embed_device)
        original_len = input_ids.shape[1]
        
        # Initialize output with input
        output_ids = input_ids.clone()
        next_token_input = input_ids
        
        # Track generation stats
        tokens_generated = 0
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get next token prediction
                outputs = model(
                    input_ids=next_token_input,
                    past_key_values=kv_cache,
                    use_cache=True
                )
                
                # Get logits for the last token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[:, indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to output sequence
                output_ids = torch.cat([output_ids, next_token], dim=1)
                
                # Update KV cache
                kv_cache = outputs.past_key_values
                
                # Prepare next input
                next_token_input = next_token
                
                tokens_generated += 1
                
                # Check for EOS token
                if next_token.item() in tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]:
                    break
        
        # Decode the generated text (skipping the input)
        generated_text = tokenizer.decode(output_ids[0, original_len:], skip_special_tokens=True)
        
        # Calculate stats
        generation_time = time.time() - start_time
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second
        }
    
    def save_kv_cache(self, kv_cache: DynamicCache, model_name: str, knowledge_id: str):
        """
        Save KV cache to disk for later reuse.
        
        Args:
            kv_cache: The KV cache to save
            model_name: Name of the model used
            knowledge_id: Identifier for the knowledge
        """
        # Create a safe filename
        safe_model_name = model_name.replace('/', '_')
        cache_path = os.path.join(self.cache_dir, f"{safe_model_name}_{knowledge_id}.pt")
        
        # Save the cache
        torch.save(kv_cache, cache_path)
        
        return cache_path
    
    def load_kv_cache(self, model_name: str, knowledge_id: str) -> Optional[Tuple[DynamicCache, int]]:
        """
        Load KV cache from disk.
        
        Args:
            model_name: Name of the model
            knowledge_id: Identifier for the knowledge
            
        Returns:
            Tuple of (kv_cache, kv_cache_length) if found, None otherwise
        """
        # Create the expected filename
        safe_model_name = model_name.replace('/', '_')
        cache_path = os.path.join(self.cache_dir, f"{safe_model_name}_{knowledge_id}.pt")
        
        # Check if the cache exists
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Load the cache
            kv_cache = torch.load(cache_path)
            
            # Get the length
            kv_len = kv_cache.key_cache[0].shape[-2]
            
            return kv_cache, kv_len
        except Exception as e:
            print(f"Error loading KV cache: {e}")
            return None
    
    def generate_knowledge_id(self, knowledge: str) -> str:
        """
        Generate a unique identifier for a knowledge text.
        
        Args:
            knowledge: The knowledge text
            
        Returns:
            Unique ID string
        """
        return hashlib.md5(knowledge.encode('utf-8')).hexdigest()[:16] 