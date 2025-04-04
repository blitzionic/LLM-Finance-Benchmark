import os
import openai
import json
from dotenv import load_dotenv
from abc import ABC, abstractmethod

load_dotenv()

class Agent(ABC):
    def __init__(self, model, function_schema, pyd_model=None):
        self.model = model
        self.pyd_model = pyd_model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPEN_AI_KEY environment variable.")
        openai.api_key = self.api_key
        self.function_schema = function_schema

    @abstractmethod
    def system_prompt(self):
        # to be implemented in each sub-agent class 
        pass

    def generate_response(self, prompt, temperature=0, max_tokens=1000, stop=None):
        try:
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": prompt}
            ]
            if self.function_schema:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    functions=[self.function_schema],
                    function_call={"name": self.function_schema.get("name")}, 
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
            else:
                print(f"Function Schema does not exist in base_agent.py")
                
            message = response.choices[0].message
            args = message.function_call.arguments
            # parses json into python dict
            parsed_args = json.loads(args)
            return parsed_args
        
        except Exception as e:
            print(f"Error generating response for Base Agent: {e}")
            return {}