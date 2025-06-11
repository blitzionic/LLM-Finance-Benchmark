import os
import json
from .base_agent import Agent
from .pyd_schema import AnswerSchema

# Initial Generator -> Reviewer -> Initial Generator (with feedback)
# The Reviewer Agent is intended to "review" the initial generated solution and provide feedback only.

class CriticReviewerAgent(Agent):
    def __init__(self, model, provider, api_key=None, topic_roles_json="topic_roles.json"):
        if os.path.exists(topic_roles_json):
            with open(topic_roles_json, 'r', encoding='utf-8') as f:
                roles = json.load(f) 
        else:
            roles = {}
            print(f"Warning: {topic_roles_json} not found. Using default role.")
            
        self.role_description = roles.get("default", "")
        # No function schema needed - just return string feedback
        super().__init__(model=model, provider=provider, function_schema=None, pyd_model=None, api_key=api_key)
    
    def system_prompt(self):
        return (
            f"{self.role_description}\n"
            "You are a strict critic focused on evaluating answers. Your role is to:\n"
            "1. Identify strengths in the reasoning and answer\n"
            "2. Point out weaknesses and gaps in the logic\n"
            "3. Provide specific suggestions for improvement\n"
            "4. Do NOT provide your own answer - focus solely on critique\n"
            "Be thorough, specific, and constructive in your feedback."
        )
    
    def review_answer(self, question, initial_answer, initial_reasoning):
        """Review the initial answer and provide detailed feedback"""
        prompt = (
            "Review the following question and answer thoroughly. Provide a detailed critique focusing on:\n"
            "1. What aspects of the answer and reasoning are strong?\n"
            "2. What are the weaknesses or gaps in the logic?\n"
            "3. What specific improvements could be made?\n\n"
            f"Original Question: {question}\n"
            f"Initial Answer: {initial_answer}\n"
            f"Initial Reasoning: {initial_reasoning}\n\n"
            "Provide your critique in a clear, constructive manner."
        )
        response = self.generate_response(prompt)
        return response