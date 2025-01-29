import os
import openai
from dotenv import load_dotenv

load_dotenv()

class Agent:
  def __init__(self, model = "gpt-4o"):
    self.model = model
    self.api_key = os.getenv("OPEN_AI_KEY")
    if not self.api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = self.api_key

  def generate_response(self, prompt, temperature = 0.7, max_tokens = 1000, stop = None):
    try: 
      response = openai.ChatCompletion.create(
        model=self.model,
        messages=[
          {"role": "system", "content": self.system_prompt()},
          {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
      )
      return response.choices[0].message('content').strip()
    except Exception as e:
      print(f"Error generating response for Base Agent: {e}")
      return ""
      
  def system_prompt(self):
        return "You are a helpful assistant."
