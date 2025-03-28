from typing import Dict, Any, Optional, List, Tuple
import time
from kv_cache_manager import KVCacheManager
from ollama_client import OllamaClient, OllamaGenerationOptions

class KVCacheGenerator:
    """
    Augmented text generator that leverages KV cache for faster inference.
    """
    
    def __init__(
        self, 
        cache_dir: str = "../cache",
        ollama_url: str = "http://localhost:11434",
        min_cache_prefix_ratio: float = 0.7
    ):
        """
        Initialize the KV Cache Generator.
        
        Args:
            cache_dir: Directory to store caches
            ollama_url: URL for Ollama API
            min_cache_prefix_ratio: Minimum prefix match ratio for cache reuse
        """
        self.cache_manager = KVCacheManager(cache_dir)
        self.ollama_client = OllamaClient(ollama_url)
        self.min_cache_prefix_ratio = min_cache_prefix_ratio
        
    def list_available_models(self) -> List[str]:
        """List models available in Ollama."""
        return self.ollama_client.list_models()
    
    def generate(
        self,
        prompt: str,
        model: str,
        options: Optional[OllamaGenerationOptions] = None,
        use_cache: bool = True,
        save_cache: bool = True,
        extra_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text with KV cache augmentation.
        
        Args:
            prompt: Input prompt
            model: Model name
            options: Generation options
            use_cache: Whether to use cached KV pairs
            save_cache: Whether to save KV cache for future use
            extra_options: Additional options for Ollama
            
        Returns:
            Dict with generation results and metrics
        """
        start_time = time.time()
        result = {
            "prompt": prompt,
            "model": model,
            "cache_hit": False,
            "cache_prefix_hit": False,
            "prefix_length": 0,
        }
        
        kv_cache = None
        
        if use_cache:
            # Try exact cache match first
            kv_cache = self.cache_manager.get_cache(prompt, model)
            if kv_cache is not None:
                result["cache_hit"] = True
            else:
                # Try prefix match
                kv_cache, prefix_length = self.cache_manager.find_best_prefix_cache(
                    prompt, model, self.min_cache_prefix_ratio
                )
                if kv_cache is not None:
                    result["cache_prefix_hit"] = True
                    result["prefix_length"] = prefix_length
        
        # Generate with or without cached KV pairs
        response_data = self.ollama_client.generate(
            prompt, model, options, extra_options, kv_cache
        )
        
        # Extract generation details
        result["text"] = response_data.get("response", "")
        result["tokens_generated"] = response_data.get("eval_count", 0)
        result["total_duration"] = response_data.get("generation_time", 0)
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
                self.cache_manager.save_cache(prompt, model, new_kv_cache)
        
        # Add total time including cache operations
        result["total_time"] = time.time() - start_time
        
        return result
    
    def benchmark(
        self, 
        prompt: str,
        model: str,
        options: Optional[OllamaGenerationOptions] = None,
        runs: int = 2,
        warm_up: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark generation with and without KV cache.
        
        Args:
            prompt: Input prompt
            model: Model name
            options: Generation options
            runs: Number of benchmark runs
            warm_up: Whether to do a warm-up run before benchmarking
            
        Returns:
            Dict with benchmark results
        """
        if warm_up:
            # Warm-up run without cache to initialize model
            print("Running warm-up...")
            self.generate(prompt, model, options, use_cache=False, save_cache=False)
        
        results = {
            "prompt": prompt,
            "model": model,
            "no_cache": [],
            "with_cache": [],
            "with_prefix_cache": []
        }
        
        # First run: no cache, but save cache
        print(f"Run 1/{runs*2}: No cache, saving cache")
        no_cache_result = self.generate(prompt, model, options, use_cache=False, save_cache=True)
        results["no_cache"].append(no_cache_result)
        
        # Second run: with full cache
        print(f"Run 2/{runs*2}: With full cache")
        cached_result = self.generate(prompt, model, options, use_cache=True, save_cache=False)
        results["with_cache"].append(cached_result)
        
        # Additional runs if requested
        for i in range(1, runs):
            # No cache run
            print(f"Run {i*2+1}/{runs*2}: No cache")
            no_cache_result = self.generate(prompt, model, options, use_cache=False, save_cache=False)
            results["no_cache"].append(no_cache_result)
            
            # With cache run
            print(f"Run {i*2+2}/{runs*2}: With cache")
            cached_result = self.generate(prompt, model, options, use_cache=True, save_cache=False)
            results["with_cache"].append(cached_result)
        
        # Benchmark with prefix cache
        if len(prompt) > 10:
            # Create a prefix prompt (75% of original)
            prefix_length = int(len(prompt) * 0.75)
            prefix_prompt = prompt[:prefix_length]
            
            # Generate and save cache for prefix
            print("Generating prefix cache...")
            self.generate(prefix_prompt, model, options, use_cache=False, save_cache=True)
            
            # Now test with the original prompt using prefix cache
            print("Testing with prefix cache...")
            prefix_result = self.generate(prompt, model, options, use_cache=True, save_cache=False)
            results["with_prefix_cache"].append(prefix_result)
        
        # Calculate averages and speedups
        self._calculate_stats(results)
        
        return results
    
    def _calculate_stats(self, results: Dict[str, Any]):
        """Calculate statistics for benchmark results."""
        # No cache stats
        if results["no_cache"]:
            no_cache_times = [r["total_duration"] for r in results["no_cache"]]
            results["no_cache_avg_time"] = sum(no_cache_times) / len(no_cache_times)
            
            no_cache_tps = [r["tokens_per_second"] for r in results["no_cache"]]
            results["no_cache_avg_tps"] = sum(no_cache_tps) / len(no_cache_tps)
        
        # Full cache stats
        if results["with_cache"]:
            cache_times = [r["total_duration"] for r in results["with_cache"]]
            results["cache_avg_time"] = sum(cache_times) / len(cache_times)
            
            cache_tps = [r["tokens_per_second"] for r in results["with_cache"]]
            results["cache_avg_tps"] = sum(cache_tps) / len(cache_tps)
            
            # Calculate speedup
            if results["no_cache_avg_time"] > 0:
                results["cache_speedup"] = results["no_cache_avg_time"] / results["cache_avg_time"]
        
        # Prefix cache stats
        if results["with_prefix_cache"]:
            prefix_times = [r["total_duration"] for r in results["with_prefix_cache"]]
            results["prefix_cache_avg_time"] = sum(prefix_times) / len(prefix_times)
            
            prefix_tps = [r["tokens_per_second"] for r in results["with_prefix_cache"]]
            results["prefix_cache_avg_tps"] = sum(prefix_tps) / len(prefix_tps)
            
            # Calculate speedup
            if results["no_cache_avg_time"] > 0:
                results["prefix_cache_speedup"] = results["no_cache_avg_time"] / results["prefix_cache_avg_time"] 