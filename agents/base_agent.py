import os
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dotenv import load_dotenv
from .llm_providers import get_llm_provider

load_dotenv()

class Agent(ABC):
    def __init__(
        self,
        model: str,
        provider: str,
        function_schema: Optional[Dict] = None,
        pyd_model = None,
        **kwargs
    ):
        """
        Initialize the agent with the specified LLM provider.
    
        Args:
            model: Model name/identifier
            provider: One of "openai", "runpod", "anthropic", "google"
            function_schema: Optional function schema for structured output
            pyd_model: Optional Pydantic model for validation
            **kwargs: Additional provider-specific arguments
        """
        self.provider = provider
        self.model = model
        self.pyd_model = pyd_model
        self.function_schema = function_schema
        self.llm = get_llm_provider(provider, model, **kwargs)

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass

    def generate_response(self, prompt: str, temperature: float = 0, max_tokens: int = 1000, stop: Optional[str] = None) -> Dict:
        """
        Generate a response using the configured LLM provider.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Optional stop sequence
            
        Returns:
            Dict containing the response
        """
        try:
            # Only include function-related parameters if function_schema is provided
            kwargs = {
                "prompt": prompt,
                "system_prompt": self.system_prompt(),
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if self.function_schema:
                kwargs["functions"] = [self.function_schema]
                kwargs["function_call"] = {"name": self.function_schema.get("name")}
            
            response = self.llm.generate_response(**kwargs)
            
            # Validate against Pydantic model if provided
            if self.pyd_model:
                return self.pyd_model(**response).dict()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {}