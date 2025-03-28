import os
import json
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import hashlib

class KVCacheManager:
    """
    Manages key-value caches for LLM generation to improve performance
    by reusing cached key-value pairs from previous runs.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the KV Cache Manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict[str, str]:
        """Load cache index from disk or create a new one."""
        index_path = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        index_path = os.path.join(self.cache_dir, "cache_index.json")
        with open(index_path, 'w') as f:
            json.dump(self.cache_index, f)
    
    def _generate_cache_key(self, prompt: str, model_name: str) -> str:
        """Generate a unique key for the prompt and model combination."""
        combined = f"{prompt}_{model_name}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def save_cache(self, prompt: str, model_name: str, kv_cache: Any):
        """
        Save KV cache for a prompt and model.
        
        Args:
            prompt: The input prompt
            model_name: Name of the model
            kv_cache: KV cache data from the model
        """
        cache_key = self._generate_cache_key(prompt, model_name)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npz")
        
        # Convert the cache to a saveable format
        if isinstance(kv_cache, dict):
            np.savez_compressed(cache_path, **kv_cache)
        else:
            np.savez_compressed(cache_path, kv_cache=kv_cache)
        
        # Update index
        self.cache_index[cache_key] = {
            "prompt": prompt, 
            "model": model_name,
            "path": cache_path,
            "timestamp": time.time()
        }
        self._save_cache_index()
        
    def get_cache(self, prompt: str, model_name: str) -> Optional[Any]:
        """
        Retrieve KV cache for a prompt and model if it exists.
        
        Args:
            prompt: The input prompt
            model_name: Name of the model
            
        Returns:
            KV cache if found, None otherwise
        """
        cache_key = self._generate_cache_key(prompt, model_name)
        if cache_key not in self.cache_index:
            return None
        
        cache_path = self.cache_index[cache_key]["path"]
        if not os.path.exists(cache_path):
            # Cache file missing, clean up index
            del self.cache_index[cache_key]
            self._save_cache_index()
            return None
        
        # Load and return the cache
        loaded = np.load(cache_path, allow_pickle=True)
        if "kv_cache" in loaded:
            return loaded["kv_cache"]
        return {k: loaded[k] for k in loaded.files}
    
    def find_best_prefix_cache(self, prompt: str, model_name: str, min_match_ratio: float = 0.7) -> Tuple[Optional[Any], int]:
        """
        Find the best prefix cache for a given prompt.
        
        Args:
            prompt: The input prompt
            model_name: Name of the model
            min_match_ratio: Minimum ratio for prefix matching
            
        Returns:
            Tuple of (kv_cache, prefix_length) if found, (None, 0) otherwise
        """
        candidates = []
        
        for cache_key, info in self.cache_index.items():
            if info["model"] != model_name:
                continue
                
            cached_prompt = info["prompt"]
            
            # Check if cached prompt is a prefix of the current prompt
            if prompt.startswith(cached_prompt):
                match_ratio = len(cached_prompt) / len(prompt)
                if match_ratio >= min_match_ratio:
                    candidates.append((cached_prompt, cache_key, match_ratio))
        
        if not candidates:
            return None, 0
        
        # Find the longest matching prefix
        best_candidate = max(candidates, key=lambda x: len(x[0]))
        best_prompt, best_key, _ = best_candidate
        
        # Load the cache
        cache_path = self.cache_index[best_key]["path"]
        if not os.path.exists(cache_path):
            # Cache file missing, clean up index
            del self.cache_index[best_key]
            self._save_cache_index()
            return None, 0
        
        # Load and return the cache
        loaded = np.load(cache_path, allow_pickle=True)
        if "kv_cache" in loaded:
            return loaded["kv_cache"], len(best_prompt)
        return {k: loaded[k] for k in loaded.files}, len(best_prompt)
    
    def clear_cache(self, days_old: Optional[int] = None):
        """
        Clear cache entries.
        
        Args:
            days_old: If provided, only clear entries older than this many days
        """
        if days_old is None:
            # Clear all cache
            for cache_key in list(self.cache_index.keys()):
                cache_path = self.cache_index[cache_key]["path"]
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            self.cache_index = {}
            self._save_cache_index()
        else:
            # Clear old cache entries
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            keys_to_delete = []
            
            for cache_key, info in self.cache_index.items():
                if info["timestamp"] < cutoff_time:
                    cache_path = info["path"]
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    keys_to_delete.append(cache_key)
            
            for key in keys_to_delete:
                del self.cache_index[key]
            
            self._save_cache_index() 