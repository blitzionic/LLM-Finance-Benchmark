from pydantic import BaseModel, validator
from openai import OpenAI

# Pydantic model for structured output
class Step(BaseModel):
    explanation: str
    output: str

class ReasoningOutput(BaseModel):
    steps: list[Step]
    final_answer: str  # Ensures only A, B, C, or D is allowed
    
    @validator("final_answer")
    def validate_final_answer(cls, value):
        if value not in ["A", "B", "C", "D"]:
            raise ValueError("final_answer must be one of A, B, C, or D.")
        return value

# Function to query GPT with Chain of Thought reasoning
def ask_gpt_with_cot(question, client, model="gpt-4o-2024-08-06"):
    """
    Queries GPT for answers with Chain of Thought reasoning.

    Parameters:
        question (str): The question formatted with multiple-choice options.
        client (OpenAI): The OpenAI client instance.
        model (str): The GPT model to use.

    Returns:
        tuple: A list of reasoning steps and the final answer.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are solving multiple-choice questions with reasoning. "
                        "Provide your reasoning step-by-step and return the answer in the following JSON format:\n\n"
                        "{\n"
                        '  "steps": [{"explanation": "<step-by-step explanation>", "output": "<intermediate output>"}],\n'
                        '  "final_answer": "<letter choice>"\n'
                        "}\n\n"
                        "The final_answer must only be one of A, B, C, or D."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )

        # Extract the response content
        
        print("Raw API Response:")
        print(completion)
        print(f"{'-'*60}")
        
        response_content = completion.choices[0].message.content

        # Strip Markdown code block markers if present
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[7:-3].strip()

        # Parse the response into the ReasoningOutput schema
        parsed_response = ReasoningOutput.parse_raw(response_content)
        print(f"Parsed Response: {parsed_response}\n{'-'*60}")
        
        print(f"Model's Final Answer: {parsed_response.final_answer}")
        print(f"{'-'*60}")

        # Return both the reasoning steps and the final answer
        return parsed_response.steps, parsed_response.final_answer
    except Exception as e:
        print(f"Error querying OpenAI API for CoT: {e}")
        try:
            print(f"Raw response content: {response_content}")
        except NameError:
            print("No response content available.")
        return [], "N"  # Return empty steps and "N" as the default final answer
