from openai import OpenAI
import json


answer_function = {
    "name": "answer_question",
    "description": "Answer a multiple-choice question",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question",
                "enum": ["A", "B", "C", "D"],  # Valid options
            },
        },
        "required": ["answer"],
    },
}

# Function for direct answer querying
def ask_gpt_direct(question, client, model):
    try:      
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are tasked with answering multiple-choice questions."},
                {"role": "user", "content": question},
            ],
            functions=[answer_function],  # Define the structured output
            function_call={"name": "answer_question"},  # Explicitly request the function
        )

        response = completion.choices[0].message  # Access the first choice
        function_call = response.function_call  # Access the function call
        args = function_call.arguments  # Extract arguments
        parsed_args = json.loads(args)  # Parse the JSON arguments
        answer = parsed_args["answer"]  # Get the "answer" key
        print("Raw API Response:", completion)
        print("Function Call:", function_call)
        print("Arguments:", args)
        return answer
    except Exception as e:
        print(f"Error in direct-answer logic: {e}")
        return "N"

    
    