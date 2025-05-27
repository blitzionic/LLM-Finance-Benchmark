from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class AnswerEnum(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

class AnswerSchema(BaseModel):
    answer: AnswerEnum = Field(
        ...,
        description="The answer to the question, one of A, B, C, or D."
    )
    feedback: str = Field(
        ...,
        description="Additional context or reasoning that explains how the answer was determined."
    )

FUNCTION_SCHEMA = {
    "name": "generate_answer",
    "description": "Generate a candidate answer along with a brief explanation. The candidate answer must be one letter among A, B, C, or D.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["A", "B", "C", "D"],
                "description": "The candidate answer, which must be one letter: A, B, C, or D."
            },
            "reasoning": {
                "type": "string",
                "description": "A brief explanation of the reasoning behind the chosen answer."
            },
            "critique": {
                "type": "string",
                "description": "A critique of the initial answer, highlighting strengths and weaknesses."
            }
        },
        "required": ["answer", "reasoning", "critique"]
    }
}

def create_function_prompt(question: str) -> str:
    return f"""You are a helpful AI assistant that always responds in valid JSON format. You must strictly follow the provided schema.

SCHEMA:
{json.dumps(FUNCTION_SCHEMA, indent=2)}

Here are some example responses:

Example 1:
{{
    "answer": "B",
    "reasoning": "The question asks about the capital of France. Paris is the capital of France, and this is option B.",
    "critique": "The answer is correct and the reasoning is clear. However, it could have provided more historical context about Paris."
}}

Example 2:
{{
    "answer": "A",
    "reasoning": "Based on the given options, option A is the most appropriate choice because...",
    "critique": "The reasoning is solid but could have considered alternative perspectives."
}}

Now, please answer the following question following the exact same format:

Question: {question}

Your response must be a valid JSON object matching the schema above. Do not include any text before or after the JSON object.

Response:"""

def parse_response(response: str) -> Optional[AnswerSchema]:
    try:
        # Find the first occurrence of { and last occurrence of }
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
            
        json_str = response[start_idx:end_idx]
        data = json.loads(json_str)
        
        # Validate required fields
        required_fields = ['answer', 'reasoning', 'critique']
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields. Found: {list(data.keys())}")
            
        # Validate answer is one of A, B, C, D
        if data['answer'] not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Invalid answer: {data['answer']}")
        
        # Create AnswerSchema instance
        return AnswerSchema(
            answer=data['answer'],
            feedback=f"Reasoning: {data['reasoning']}\nCritique: {data['critique']}"
        )
    except Exception as e:
        print(f"Error parsing response: {e}")
        print("\nRaw response:")
        print(response)
        return None

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Test prompt
question = "What is the capital of France?"
prompt = create_function_prompt(question)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response with more tokens to ensure complete JSON
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1
)

# Decode and parse response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
parsed_response = parse_response(response)

if parsed_response:
    print("\nParsed Response:")
    print(f"Answer: {parsed_response.answer}")
    print(f"Feedback: {parsed_response.feedback}")
else:
    print("\nRaw Response:")
    print(response)
