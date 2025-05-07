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
        "required": ["answer", "reasoning"]
    }
}

class DeciderAgent(Agent):
    def __init__(self, topic, model="gpt-4o-mini", topic_roles_json="config/topic_roles.json"):
        # Optionally load a role description if needed; here we use a default.
        self.role_description = "You are an analytical decision-maker with expertise in financial reasoning."
        super().__init__(model=model, pyd_model=AnswerSchema)
        self.function_schema = FUNCTION_SCHEMA
    
    def system_prompt(self):
        return (
            f"{self.role_description}\n"
            "You are the final decision-maker. Review the candidate answers and reasonings from the inital, reviewer, challenger, and refiner agents. "
            "Evaluate the strengths and weaknesses of each response and select the final best answer and provide your justification. "
            "Select the best final answer (A, B, C, or D) and your reasoning."
        )
    
    def process(self, question, initial_answer, initial_reasoning, reviewer_answer, reviewer_reasoning, challenger_answer,
                challenger_reasoning, refiner_answer, refiner_reasoning):
        prompt = (
            "Based on the following question and each question/reasoning, synthesize the strengths of each persecptive into a single "
            "refined answer and reasoning.\n"
            f"Original Question: {question}\n"
            f"Initial Answer: '{initial_answer}', Initial Reasoning: '{initial_reasoning}'\n"
            f"Reviewer Answer: '{reviewer_answer}, Reviewer Reasoning: '{reviewer_reasoning}'\n"
            f"Challenger Answer: '{challenger_answer}', Challenger Reasoning: '{challenger_reasoning}'\n"
            f"Refiner Answer: '{refiner_answer}', Refiner Reasoning: '{refiner_reasoning}'"
        )
        response = self.generate_response(prompt)
        return response 
