import os
import time
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
from enum import Enum

# Import our modules
from hf_cache_manager import HFCacheManager
from kv_cache_manager import KVCacheManager
from ollama_client import OllamaClient, OllamaGenerationOptions

class ModelBackend(Enum):
    """Enum for supported model backends."""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

class CAGGenerator:
    """
    Cache-Augmented Generation (CAG) system that works with both HuggingFace and Ollama models.
    """
    
    def __init__(
        self,
        backend: ModelBackend = ModelBackend.OLLAMA,
        hf_model=None,
        hf_tokenizer=None,
        ollama_url: str = "http://localhost:11434",
        cache_dir: str = "../cache"
    ):
        """
        Initialize the CAG Generator.
        
        Args:
            backend: Which model backend to use (HuggingFace or Ollama)
            hf_model: HuggingFace model (required if using HF backend)
            hf_tokenizer: HuggingFace tokenizer (required if using HF backend)
            ollama_url: URL for Ollama API (for Ollama backend)
            cache_dir: Base directory for caches
        """
        self.backend = backend
        self.cache_dir = cache_dir
        
        # Initialize backend-specific components
        if backend == ModelBackend.HUGGINGFACE:
            if hf_model is None or hf_tokenizer is None:
                raise ValueError("HuggingFace model and tokenizer must be provided when using HF backend")
            self.model = hf_model
            self.tokenizer = hf_tokenizer
            self.hf_cache_manager = HFCacheManager(os.path.join(cache_dir, "hf_cache"))
            self.current_kv_cache = None
            self.current_kv_cache_len = 0
            self.current_knowledge_id = None
        else:  # Ollama backend
            self.ollama_client = OllamaClient(ollama_url)
            self.kv_cache_manager = KVCacheManager(os.path.join(cache_dir, "ollama_cache"))
    
    def list_available_models(self) -> List[str]:
        """
        List available models for the current backend.
        
        Returns:
            List of model names
        """
        if self.backend == ModelBackend.HUGGINGFACE:
            # For HF, just return the current model name
            return [self.model.config.name_or_path]
        else:
            # For Ollama, query available models
            return self.ollama_client.list_models()
    
    def preload_knowledge(
        self,
        knowledge: str, 
        model_name: str,
        system_prompt: Optional[str] = None,
        save_to_disk: bool = True
    ) -> str:
        """
        Preload knowledge into KV cache.
        
        Args:
            knowledge: Text knowledge to preload
            model_name: Model name (important for Ollama backend)
            system_prompt: Optional system instructions
            save_to_disk: Whether to save the KV cache to disk
            
        Returns:
            Knowledge ID for future reference
        """
        if self.backend == ModelBackend.HUGGINGFACE:
            # Generate knowledge ID
            knowledge_id = self.hf_cache_manager.generate_knowledge_id(knowledge)
            
            # Try to load from disk first
            loaded_cache = self.hf_cache_manager.load_kv_cache(model_name, knowledge_id)
            
            if loaded_cache:
                self.current_kv_cache, self.current_kv_cache_len = loaded_cache
                self.current_knowledge_id = knowledge_id
                print(f"Loaded knowledge KV cache from disk (length: {self.current_kv_cache_len})")
            else:
                # Generate new KV cache
                start_time = time.time()
                default_system = "You are a helpful assistant that answers questions based on the provided knowledge."
                system_prompt = system_prompt or default_system
                
                self.current_kv_cache, self.current_kv_cache_len = self.hf_cache_manager.preprocess_knowledge(
                    self.model, self.tokenizer, knowledge, system_prompt
                )
                
                self.current_knowledge_id = knowledge_id
                
                print(f"Generated knowledge KV cache in {time.time() - start_time:.2f}s (length: {self.current_kv_cache_len})")
                
                # Save to disk if requested
                if save_to_disk:
                    self.hf_cache_manager.save_kv_cache(self.current_kv_cache, model_name, knowledge_id)
                    print(f"Saved knowledge KV cache to disk")
            
            return knowledge_id
        else:
            # For Ollama, we don't preload knowledge in the same way
            # Instead, we'll use the document content for RAG-style augmentation
            return ""
    
    def generate(
        self,
        prompt: str,
        model_name: str,
        use_cache: bool = True,
        save_cache: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        knowledge_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text with Cache Augmentation.
        
        Args:
            prompt: Input prompt
            model_name: Model name
            use_cache: Whether to use cached KV pairs
            save_cache: Whether to save KV cache for future use
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            knowledge_text: Optional knowledge text for RAG-style augmentation
            
        Returns:
            Dictionary with generation results
        """
        start_time = time.time()
        
        if self.backend == ModelBackend.HUGGINGFACE:
            if self.current_kv_cache is None:
                if knowledge_text:
                    # Preload knowledge if provided
                    self.preload_knowledge(knowledge_text, model_name)
                else:
                    raise ValueError("No knowledge has been preloaded and no knowledge_text provided")
            
            # Generate with HF model using preloaded KV cache
            result = self.hf_cache_manager.generate_with_knowledge(
                self.model,
                self.tokenizer,
                prompt,
                self.current_kv_cache,
                self.current_kv_cache_len,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Add metadata
            result["prompt"] = prompt
            result["model"] = model_name
            result["total_time"] = time.time() - start_time
            result["cache_hit"] = True  # Always using the preloaded cache
            
            return result
        else:
            # For Ollama backend
            options = OllamaGenerationOptions(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            # If knowledge text is provided, prepend it to the prompt
            if knowledge_text:
                augmented_prompt = f"Knowledge:\n{knowledge_text}\n\nQuestion: {prompt}"
            else:
                augmented_prompt = prompt
            
            # Use our existing KV cache generation logic
            if use_cache:
                # Try to get an exact cache match
                kv_cache = self.kv_cache_manager.get_cache(augmented_prompt, model_name) if use_cache else None
                
                result = {
                    "prompt": prompt,
                    "model": model_name,
                    "cache_hit": kv_cache is not None,
                    "cache_prefix_hit": False,
                    "prefix_length": 0,
                    "knowledge_augmented": knowledge_text is not None
                }
                
                if kv_cache is None and use_cache:
                    # Try to find a prefix match
                    kv_cache, prefix_length = self.kv_cache_manager.find_best_prefix_cache(
                        augmented_prompt, model_name, min_match_ratio=0.7
                    )
                    if kv_cache is not None:
                        result["cache_prefix_hit"] = True
                        result["prefix_length"] = prefix_length
            else:
                kv_cache = None
                result = {
                    "prompt": prompt,
                    "model": model_name,
                    "cache_hit": False,
                    "cache_prefix_hit": False,
                    "prefix_length": 0,
                    "knowledge_augmented": knowledge_text is not None
                }
            
            # Generate with Ollama
            response_data = self.ollama_client.generate(
                augmented_prompt, model_name, options, kv_cache=kv_cache
            )
            
            # Extract generation details
            result["text"] = response_data.get("response", "")
            result["tokens_generated"] = response_data.get("eval_count", 0)
            result["generation_time"] = response_data.get("generation_time", 0)
            result["tokens_per_second"] = response_data.get("tokens_per_second", 0)
            
            # Save cache if requested
            if save_cache and "error" not in response_data:
                # Extract KV cache from the response
                new_kv_cache = None
                if "kv_cache" in response_data:
                    new_kv_cache = response_data["kv_cache"]
                elif "final_kv_cache" in response_data:
                    new_kv_cache = response_data["final_kv_cache"]
                    
                if new_kv_cache is not None:
                    self.kv_cache_manager.save_cache(augmented_prompt, model_name, new_kv_cache)
            
            # Add total time
            result["total_time"] = time.time() - start_time
            
            return result
    
    def benchmark_cag(
        self,
        query: str,
        model_name: str,
        knowledge_text: str,
        runs: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Benchmark CAG performance with and without preloaded knowledge.
        
        Args:
            query: Test query
            model_name: Model name
            knowledge_text: Knowledge text to use
            runs: Number of benchmark runs
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "query": query,
            "model": model_name,
            "knowledge_size": len(knowledge_text),
            "standard_approach": [],
            "cag_approach": []
        }
        
        if self.backend == ModelBackend.HUGGINGFACE:
            # First approach: Standard generation with knowledge appended to prompt
            print("Testing standard approach (knowledge in prompt)...")
            standard_prompt = f"Knowledge:\n{knowledge_text}\n\nQuestion: {query}"
            
            for i in range(runs):
                print(f"Run {i+1}/{runs} for standard approach")
                start_time = time.time()
                
                # Tokenize
                input_ids = self.tokenizer.encode(standard_prompt, return_tensors="pt").to(self.model.device)
                
                # Generate without KV cache
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0
                    )
                
                # Get generated text
                generated_text = self.tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
                
                # Calculate stats
                gen_time = time.time() - start_time
                results["standard_approach"].append({
                    "text": generated_text,
                    "time": gen_time,
                    "run": i+1
                })
            
            # Second approach: CAG with preloaded knowledge
            print("Testing CAG approach (knowledge in KV cache)...")
            
            # Preload knowledge
            knowledge_id = self.preload_knowledge(knowledge_text, model_name)
            
            for i in range(runs):
                print(f"Run {i+1}/{runs} for CAG approach")
                
                # Generate with preloaded KV cache
                result = self.generate(
                    prompt=query,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                
                results["cag_approach"].append({
                    "text": result["text"],
                    "time": result["generation_time"],
                    "run": i+1
                })
        else:
            # For Ollama, we'll test both with normal generation and with cache reuse
            
            # First approach: Standard generation with knowledge appended to prompt
            print("Testing standard approach (knowledge in prompt, no cache)...")
            
            for i in range(runs):
                print(f"Run {i+1}/{runs} for standard approach")
                
                # Generate without using cache
                result = self.generate(
                    prompt=query,
                    model_name=model_name,
                    use_cache=False,
                    save_cache=False,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    knowledge_text=knowledge_text
                )
                
                results["standard_approach"].append({
                    "text": result["text"],
                    "time": result["generation_time"],
                    "run": i+1
                })
            
            # Second approach: Preload knowledge in cache and reuse
            print("Testing CAG approach (knowledge with cache reuse)...")
            
            # First run to create the cache
            print("Initial run to create cache...")
            self.generate(
                prompt=query,
                model_name=model_name,
                use_cache=False,
                save_cache=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                knowledge_text=knowledge_text
            )
            
            for i in range(runs):
                print(f"Run {i+1}/{runs} for CAG approach")
                
                # Generate with cache reuse
                result = self.generate(
                    prompt=query,
                    model_name=model_name,
                    use_cache=True,
                    save_cache=False,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    knowledge_text=knowledge_text
                )
                
                results["cag_approach"].append({
                    "text": result["text"],
                    "time": result["generation_time"],
                    "run": i+1
                })
        
        # Calculate average times
        if results["standard_approach"]:
            std_times = [r["time"] for r in results["standard_approach"]]
            results["standard_avg_time"] = sum(std_times) / len(std_times)
        
        if results["cag_approach"]:
            cag_times = [r["time"] for r in results["cag_approach"]]
            results["cag_avg_time"] = sum(cag_times) / len(cag_times)
            
            # Calculate speedup
            if "standard_avg_time" in results:
                results["speedup"] = results["standard_avg_time"] / results["cag_avg_time"]
        
        return results 