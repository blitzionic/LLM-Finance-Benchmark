import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import openai
from openai import OpenAI as OpenAIClient
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(ABC):
    
    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 1000,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Dict] = None
    ) -> Dict:
        """Generate a response from the LLM"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = OpenAIClient(api_key=api_key)
    
    # called in each agent class
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0,
        max_tokens: int = 1000, functions: Optional[List[Dict]] = None, function_call: Optional[Dict] = None) -> Dict: 
        # returns a dictionary with the answer and the reasoning 
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if functions:
            kwargs["functions"] = functions
        if function_call:
            kwargs["function_call"] = function_call
            
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        
        # Handle function call response
        if hasattr(message, 'function_call') and message.function_call is not None:
            try:
                return json.loads(message.function_call.arguments)
            except (json.JSONDecodeError, AttributeError):
                return {"answer": message.content}
        
        # Handle regular text response
        return {"answer": message.content}

class RunPodLlamaProvider(LLMProvider):
    """RunPod-hosted Llama model provider"""
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-2-70b-chat-hf",
        runpod_api_key: Optional[str] = None,
        runpod_endpoint: Optional[str] = None
    ):
        self.model = model
        self.api_key = runpod_api_key or os.getenv("RUNPOD_API_KEY")
        self.endpoint = runpod_endpoint or os.getenv("RUNPOD_ENDPOINT")
        if not self.api_key or not self.endpoint:
            raise ValueError("RunPod API key and endpoint must be provided")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0, max_tokens: int = 1000,
        functions: Optional[List[Dict]] = None, function_call: Optional[Dict] = None) -> Dict:
        # Format the prompt according to Llama chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            raise Exception(f"RunPod API error: {result['error']}")
            
        # Parse the response
        content = result["choices"][0]["message"]["content"]
        
        # If functions were requested, try to parse the response as JSON
        if functions:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If parsing fails, return the raw content
                return {"answer": content}
        
        return {"answer": content}

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, model: str = "claude-3-opus-20240229"):
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 1000,
        functions: Optional[List[Dict]] = None,
        function_call: Optional[Dict] = None
    ) -> Dict:
        from anthropic import Anthropic
        
        client = Anthropic(api_key=self.api_key)
        
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = message.content[0].text
        
        # If functions were requested, try to parse the response as JSON
        if functions:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"answer": content}
        
        return {"answer": content}

class GoogleProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, model: str = "gemini-pro", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0, max_tokens: int = 1000,
        functions: Optional[List[Dict]] = None, function_call: Optional[Dict] = None) -> Dict:
        import google.generativeai as genai
        import re
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        
        # If functions are provided, modify the prompt to request structured output
        if functions:
            function_schema = functions[0]  # We only use one function schema
            full_prompt = (
                f"{system_prompt}\n\n{prompt}\n\n"
                "IMPORTANT: Your response must be in this exact format:\n"
                "{\n"
                '  "answer": "A",  # Must be exactly one of: A, B, C, or D\n'
                '  "reasoning": "Your explanation here"\n'
                "}\n"
                "Do not include any other text or formatting."
            )
        else:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        
        content = response.text
        
        # If functions were requested, try to parse the response as JSON
        if functions:
            try:
                # First try to find JSON in the response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    # Ensure answer is one of A, B, C, D
                    if parsed.get("answer") in ["A", "B", "C", "D"]:
                        return parsed
            except (json.JSONDecodeError, AttributeError):
                pass
            
            # If JSON parsing fails, try to extract just the answer letter
            answer_match = re.search(r'[A-D]', content)
            if answer_match:
                return {
                    "answer": answer_match.group(0),
                    "reasoning": content
                }
            
            # If all else fails, return the raw content
            return {"answer": content}
        
        return {"answer": content}

def get_llm_provider(provider: str, model: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Returns the corresponding LLM provider defined above 
    
    provider: One of "openai", "runpod", "anthropic", "google"
    model: Model name/identifier
    **kwargs: Additional provider-specific arguments
    
    Returns:
        An instance of the requested LLM provider
    """
    providers = {
        "openai": OpenAIProvider,
        "runpod": RunPodLlamaProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")   
    
    return providers[provider](model=model, **kwargs) 