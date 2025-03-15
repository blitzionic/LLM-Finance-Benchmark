'''
Initial Generator Agent -> Reviewer Agent -> Challenger Agent -> Refiner Agent -> (Decider Agent)
The decider agent aggregates candidate answers and their reasoning, and selects the optimal final answer
'''

import os
import json
from base_agent import Agent
from answer_schema import AnswerSchema

FUNCTION_SCHEMA = {
    "name": "decide_answer",
    "description": "Select the best candidate answer from multiple agents and provide a brief evaluation summary. The final answer must be one letter among A, B, C, or D.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["A", "B", "C", "D"],
                "description": "The final selected answer, one letter: A, B, C, or D."
            },
            "feedback": {
                "type": "string",
                "description": "A brief explanation summarizing why this answer was chosen."
            }
        },
        "required": ["answer", "feedback"]
    }
}

class DeciderAgent(Agent):
    def __init__(self, model="gpt-4o-mini", topic_roles_json="config/topic_roles.json"):
        # Optionally load a role description if needed; here we use a default.
        self.role_description = "You are an analytical decision-maker with expertise in financial reasoning."
        super().__init__(model=model, pyd_model=AnswerSchema)
        self.function_schema = FUNCTION_SCHEMA
    
    def system_prompt(self):
        return (
            f"{self.role_description}\n"
            "You have received candidate answers from several agents, each with their chosen answer and supporting reasoning. "
            "Your task is to review all candidate responses, compare their strengths and weaknesses, and select the best final answer for the question. "
            "Return your final decision as a JSON object with two keys: 'answer' (one letter: A, B, C, or D) and 'feedback' (a brief explanation of your decision)."
        )
    
    def process(self, candidate_responses):
        """
        candidate_responses: a list of dictionaries, each containing keys "answer" and "feedback"
        """
        # Format the candidate responses for the prompt.
        responses_str = "\n".join(
            [f"Candidate {i+1}: Answer: {resp.get('answer')}, Feedback: {resp.get('feedback')}"
             for i, resp in enumerate(candidate_responses)]
        )
        
        prompt = (
            "Below are the candidate answers provided by different agents:\n"
            f"{responses_str}\n\n"
            "Please analyze these responses and select the best final answer. "
            "Return your final decision as a JSON object with two keys: 'answer' and 'feedback'."
        )
        response = self.generate_response(prompt)
        return response.get("answer", "")
