import os
import openai
import json 
from dotenv import load_dotenv
from abc import ABC, abstractmethod

load_dotenv()

class Agent:
  def __init__(self, model = "gept-4o", response_model = None):
    self.model = model
    self.pyd_reponse_model = response_model
    self.api_key = os.getenv("OPEN_AI_KEY")
    if not self.api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = self.api_key
  
  @abstractmethod
  def system_prompt(selfn_schema = None):
    # returns system prompt for the agent - to be implemented in agent classes
    pass

  def generate_response(self, prompt, temperature = 0, max_tokens = 1000, stop = None):
    try: 
      messages=[
        {"role": "system", "content": self.system_prompt()},
        {"role": "user", "content": prompt}
      ],
      
      if self.function_shema:
        response = openai.ChatCompletion.create(
          model = self.model,
          messages = messages, 
          response_format = self.pyd_response_model
          temperature = temperature,
          max_tokens = max_tokens,
          stop = stop,
        )
        message = response.choices[0].message
        return message.parsed.dict() 
        
      else:
        print("not using defined pydantic model in base_agent.py")
        response = openai.ChatCompletion.create(
          model=self.model,
          messages=messages,
          temperature=temperature,
          max_tokens=max_tokens,
          stop=stop,
        )
        
        return response.choices[0].message['content'].strip()
    
    except Exception as e:
      print(f"Error generating response for Base Agent: {e}")
      return {} 
