# Multi-Provider LLM Client with Free API Fallbacks

"""
Universal LLM client supporting multiple free providers:
1. Ollama (local - fastest, no API key needed)
2. Groq (free tier - 1000 req/day, very fast)
3. HuggingFace Inference (free - 300 req/hr)
4. Fallback to deterministic responses

No API keys required for basic functionality!
"""

import os
import requests
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Import config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, LLAMA_MODEL, LLM_TIMEOUT


@dataclass
class LLMProvider:
    """LLM Provider configuration."""
    name: str
    base_url: str
    model: str
    requires_key: bool
    api_key_env: str = ""


# Available providers (in priority order)
PROVIDERS = [
    LLMProvider("ollama", OLLAMA_BASE_URL, LLAMA_MODEL, False),
    LLMProvider("groq", "https://api.groq.com/openai/v1", "llama-3.1-8b-instant", True, "GROQ_API_KEY"),
    LLMProvider("huggingface", "https://api-inference.huggingface.co/models", "mistralai/Mistral-7B-Instruct-v0.2", True, "HF_API_KEY"),
]


class UniversalLLMClient:
    """
    Universal LLM client that tries multiple providers.
    Automatically falls back if one provider fails.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.active_provider: Optional[LLMProvider] = None
        self._cache: Dict[str, str] = {}  # Simple response cache
        
        # Coaching-specific responses for fallback
        self.coaching_responses = {
            "improve": "Focus on timing! Watch the gap center and flap rhythmically.",
            "die": "You crashed because of positioning. Aim for the middle of gaps.",
            "tip": "Pro tip: The best players use fewer flaps. Quality over quantity!",
            "help": "I'm your AI coach! Ask me about improving your gameplay.",
            "default": "Keep practicing! Every game makes you better. 🎮"
        }
        
    def detect_provider(self) -> Optional[LLMProvider]:
        """Detect which LLM provider is available."""
        if self.active_provider:
            return self.active_provider
            
        for provider in PROVIDERS:
            if self._test_provider(provider):
                self.active_provider = provider
                print(f"✅ LLM Provider: {provider.name} ({provider.model})")
                return provider
                
        print("⚠️ No LLM provider available - using fallback responses")
        return None
        
    def _test_provider(self, provider: LLMProvider) -> bool:
        """Test if a provider is accessible."""
        try:
            if provider.name == "ollama":
                resp = self.session.get(f"{provider.base_url}/api/tags", timeout=3)
                return resp.status_code == 200
                
            elif provider.name == "groq":
                api_key = os.environ.get(provider.api_key_env, "")
                if not api_key:
                    return False
                resp = self.session.get(
                    f"{provider.base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=5
                )
                return resp.status_code == 200
                
            elif provider.name == "huggingface":
                api_key = os.environ.get(provider.api_key_env, "")
                if not api_key:
                    return False
                return True  # HuggingFace doesn't have a free health check
                
        except Exception:
            pass
        return False
        
    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        return self.detect_provider() is not None
        
    def generate(self, prompt: str, system: str = "", 
                 temperature: float = 0.7, max_tokens: int = 200) -> str:
        """
        Generate a response using the best available provider.
        
        Falls back gracefully through providers and ultimately to
        deterministic responses.
        """
        # Check cache first
        cache_key = f"{prompt[:50]}:{system[:20]}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        provider = self.detect_provider()
        
        if provider is None:
            return self._get_coaching_fallback(prompt)
            
        try:
            if provider.name == "ollama":
                response = self._generate_ollama(prompt, system, temperature, max_tokens)
            elif provider.name == "groq":
                response = self._generate_groq(prompt, system, temperature, max_tokens)
            elif provider.name == "huggingface":
                response = self._generate_huggingface(prompt, temperature, max_tokens)
            else:
                response = self._get_coaching_fallback(prompt)
                
            # Cache successful responses
            if response and not response.startswith("Error"):
                self._cache[cache_key] = response
                
            return response
            
        except Exception as e:
            print(f"LLM Error ({provider.name}): {e}")
            # Try next provider
            self.active_provider = None
            return self._get_coaching_fallback(prompt)
            
    def _generate_ollama(self, prompt: str, system: str, 
                         temperature: float, max_tokens: int) -> str:
        """Generate using local Ollama."""
        data = {
            "model": self.active_provider.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        resp = self.session.post(
            f"{self.active_provider.base_url}/api/generate",
            json=data,
            timeout=LLM_TIMEOUT
        )
        
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return f"Error: {resp.status_code}"
        
    def _generate_groq(self, prompt: str, system: str,
                       temperature: float, max_tokens: int) -> str:
        """Generate using Groq API (OpenAI compatible)."""
        api_key = os.environ.get("GROQ_API_KEY", "")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        resp = self.session.post(
            f"{self.active_provider.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.active_provider.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=LLM_TIMEOUT
        )
        
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        return f"Error: {resp.status_code}"
        
    def _generate_huggingface(self, prompt: str, 
                              temperature: float, max_tokens: int) -> str:
        """Generate using HuggingFace Inference API."""
        api_key = os.environ.get("HF_API_KEY", "")
        
        resp = self.session.post(
            f"{self.active_provider.base_url}/{self.active_provider.model}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False
                }
            },
            timeout=LLM_TIMEOUT
        )
        
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
        return f"Error: {resp.status_code}"
        
    def _get_coaching_fallback(self, prompt: str) -> str:
        """Get intelligent fallback response based on prompt context."""
        prompt_lower = prompt.lower()
        
        for keyword, response in self.coaching_responses.items():
            if keyword in prompt_lower:
                return response
                
        return self.coaching_responses["default"]
        
    def chat(self, messages: List[Dict[str, str]], 
             temperature: float = 0.7, max_tokens: int = 200) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {'role': 'user/assistant/system', 'content': '...'}
        """
        # Convert to single prompt for providers that don't support chat
        if not messages:
            return self._get_coaching_fallback("")
            
        # Build combined prompt
        prompt_parts = []
        system = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
                
        prompt = "\n".join(prompt_parts)
        if prompt:
            prompt += "\nAssistant:"
            
        return self.generate(prompt, system, temperature, max_tokens)
        
    def get_provider_info(self) -> Dict[str, Any]:
        """Get info about active provider."""
        provider = self.detect_provider()
        if provider:
            return {
                "name": provider.name,
                "model": provider.model,
                "status": "active"
            }
        return {
            "name": "fallback",
            "model": "deterministic",
            "status": "no LLM available"
        }


# Singleton instance
_client: Optional[UniversalLLMClient] = None


def get_universal_client() -> UniversalLLMClient:
    """Get singleton universal LLM client."""
    global _client
    if _client is None:
        _client = UniversalLLMClient()
    return _client


# Test the client
if __name__ == "__main__":
    client = get_universal_client()
    print(f"Provider: {client.get_provider_info()}")
    
    response = client.generate(
        "Give me a quick tip for Flappy Bird",
        system="You are a Flappy Bird coach. Be brief and helpful."
    )
    print(f"Response: {response}")
