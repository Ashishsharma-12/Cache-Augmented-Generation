import requests
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import time
from pydantic import BaseModel, Field

class OllamaGenerationOptions(BaseModel):
    """Options for Ollama text generation."""
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0)
    max_tokens: int = Field(2048, ge=1)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    stop: List[str] = Field(default_factory=list)
    stream: bool = Field(False)

class OllamaClient:
    """Client for interacting with Ollama local API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API
        """
        self.base_url = base_url
        self.api_generate = f"{base_url}/api/generate"
        self.api_embeddings = f"{base_url}/api/embeddings"
        self.api_tags = f"{base_url}/api/tags"
    
    def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(self.api_tags)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(
        self, 
        prompt: str, 
        model: str,
        options: Optional[OllamaGenerationOptions] = None,
        raw_options: Optional[Dict[str, Any]] = None,
        kv_cache: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Input prompt
            model: Model name
            options: Generation options
            raw_options: Additional raw options to pass directly to Ollama
            kv_cache: Optional KV cache to use for generation
            
        Returns:
            Response dictionary containing generated text and metadata
        """
        if options is None:
            options = OllamaGenerationOptions()
        
        payload = {
            "model": model,
            "prompt": prompt,
            **options.dict(exclude_none=True)
        }
        
        # Add any raw options
        if raw_options:
            payload.update(raw_options)
        
        # Add KV cache if provided
        if kv_cache is not None:
            kv_cache_options = {"raw": True}
            # Different models may have different KV cache formats
            # We adapt based on the format provided
            if isinstance(kv_cache, dict):
                # Add the appropriate KV cache structure based on what's in the dict
                # This part is model-specific and may need adjustment
                if "prefill" in kv_cache and "kv_cache" in kv_cache:
                    # For models that separate prefill and KV cache
                    kv_cache_options["prefill"] = kv_cache["prefill"]
                    kv_cache_options["kv_cache"] = kv_cache["kv_cache"]
                else:
                    # For models with a unified KV cache structure
                    kv_cache_options["kv_cache"] = kv_cache
            else:
                # Assume the kv_cache is in the correct format
                kv_cache_options["kv_cache"] = kv_cache
                
            payload["options"] = {**payload.get("options", {}), **kv_cache_options}
        
        try:
            start_time = time.time()
            response = requests.post(self.api_generate, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Calculate generation speed
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_generated = data.get("eval_count", 0)
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            # Add timing and speed metadata
            data["generation_time"] = generation_time
            data["tokens_per_second"] = tokens_per_second
            
            return data
        except Exception as e:
            print(f"Error generating text: {e}")
            return {"error": str(e)}
    
    def generate_with_kv_cache(
        self, 
        prompt: str, 
        model: str,
        options: Optional[OllamaGenerationOptions] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """
        Generate text and return both the text and KV cache for reuse.
        
        Args:
            prompt: Input prompt
            model: Model name
            options: Generation options
            extra_options: Additional options to pass to Ollama
            
        Returns:
            Tuple of (generated_text, kv_cache)
        """
        # Ensure we request the KV cache in the response
        if extra_options is None:
            extra_options = {}
        
        # Raw mode needs to be enabled to access kv cache
        raw_options = {
            "options": {
                "raw": True,
                **extra_options.get("options", {})
            }
        }
        
        response = self.generate(prompt, model, options, raw_options)
        
        if "error" in response:
            return response["error"], None
        
        # Extract KV cache from response
        kv_cache = None
        if "kv_cache" in response:
            kv_cache = response["kv_cache"]
        elif "final_kv_cache" in response:
            kv_cache = response["final_kv_cache"]
        
        return response.get("response", ""), kv_cache 