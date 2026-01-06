# Ollama Client - Local LLM/Vision API

"""
Client for Ollama API enabling local inference with Llama and LLaVA models.
Supports both text generation and vision-language tasks.
"""

import requests
import json
import base64
from typing import Optional, Dict, Any, Generator, List
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, LLAMA_MODEL, LLAVA_MODEL, LLM_TIMEOUT


class OllamaClient:
    """
    Client for local Ollama API.
    
    Supports:
    - Text generation with Llama models
    - Vision-language generation with LLaVA
    - Streaming responses
    - Chat completion format
    """
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip('/')
        self._session = None
        
    @property
    def session(self) -> requests.Session:
        """Get or create requests session."""
        if self._session is None:
            self._session = requests.Session()
        return self._session
        
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
            
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            pass
        return []
        
    def has_model(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        models = self.list_models()
        return any(model_name in m for m in models)
        
    def generate(self, 
                 prompt: str, 
                 model: str = LLAMA_MODEL,
                 system: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 stream: bool = False) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: User prompt
            model: Model name (default: llama3.2)
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text
        """
        data = {
            'model': model,
            'prompt': prompt,
            'stream': stream,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        if system:
            data['system'] = system
            
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                if stream:
                    return self._handle_stream(response)
                result = response.json()
                return result.get('response', '')
            else:
                return f"Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out"
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
            
    def chat(self,
             messages: List[Dict[str, str]],
             model: str = LLAMA_MODEL,
             temperature: float = 0.7,
             max_tokens: int = 256) -> str:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {'role': 'user/assistant/system', 'content': '...'}
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Assistant's response
        """
        data = {
            'model': model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=data,
                timeout=LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                return f"Error: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
            
    def generate_with_image(self,
                            prompt: str,
                            image_path: str,
                            model: str = LLAVA_MODEL,
                            temperature: float = 0.7,
                            max_tokens: int = 256) -> str:
        """
        Generate text with image input (vision-language).
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            model: Vision model name (default: llava)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Generated text describing/analyzing the image
        """
        # Read and encode image
        image_path = Path(image_path)
        if not image_path.exists():
            return f"Error: Image not found: {image_path}"
            
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
            
        data = {
            'model': model,
            'prompt': prompt,
            'images': [image_data],
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=LLM_TIMEOUT * 2  # Vision takes longer
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return f"Error: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
            
    def generate_with_image_bytes(self,
                                   prompt: str,
                                   image_bytes: bytes,
                                   model: str = LLAVA_MODEL,
                                   temperature: float = 0.7,
                                   max_tokens: int = 256) -> str:
        """
        Generate text with image bytes input.
        
        Args:
            prompt: Text prompt
            image_bytes: Raw image bytes
            model: Vision model name
            temperature: Sampling temperature
            max_tokens: Max tokens
            
        Returns:
            Generated text
        """
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        data = {
            'model': model,
            'prompt': prompt,
            'images': [image_data],
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=LLM_TIMEOUT * 2
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return f"Error: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
            
    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get('response', '')
                
    def close(self):
        """Close the session."""
        if self._session:
            self._session.close()
            self._session = None


# Singleton instance
_client: Optional[OllamaClient] = None


def get_client() -> OllamaClient:
    """Get singleton Ollama client."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
