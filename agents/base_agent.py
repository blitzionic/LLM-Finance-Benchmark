import os
from abc import ABC, abstractmethod
from typing import Dict, Optional
from dotenv import load_dotenv
from .llm_providers import get_llm_provider

load_dotenv()

class Agent(ABC):
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        function_schema: Optional[Dict] = None,
        pyd_model = None,
        **kwargs
    ):
        """
        Initialize the agent with the specified LLM provider.
    
        Args:
            provider: One of "openai", "runpod", "anthropic", "google"
            model: Model name/identifier
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

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 1000,
        stop: Optional[str] = None
    ) -> Dict:
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
            response = self.llm.generate_response(
                prompt=prompt,
                system_prompt=self.system_prompt(),
                temperature=temperature,
                max_tokens=max_tokens,
                functions=[self.function_schema] if self.function_schema else None,
                function_call={"name": self.function_schema.get("name")} if self.function_schema else None
            )
            
            # Validate against Pydantic model if provided
            if self.pyd_model:
                return self.pyd_model(**response).dict()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {}