import os
import json
from base_agent import Agent
from answer_schema import AnswerSchema
from .llm_providers import get_llm_provider

# Intitial Generator -> Reviewer -> Challenger Agent 
# The Reviewer Agent is intended to "review" the initial generated solution. 
# It reviews the solution returns its own answer. 

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

class CriticReviewerAgent(Agent):
    def __init__(self, topic, model="gpt-4o-mini", topic_roles_json="topic_roles.json"):
        if os.path.exists(topic_roles_json):
            with open(topic_roles_json, 'r', encoding='utf-8') as f:
                roles = json.load(f) 
        else:
            roles = {}
            print(f"Warning: {topic_roles_json} not found. Using default role.")
            
        self.topic = topic 
        self.role_description = roles.get(topic.lower(), "")
        super().__init__(model=model, function_schema=FUNCTION_SCHEMA, pyd_model=AnswerSchema)
    
    def system_prompt(self):
        return (
            f"{self.role_description}\n"
            "You are a strict critic. Evaluate the given solution for accuracy and correctness. "
            "Identify any gaps in reasoning or potential errors, then select the correct answer (A, B, C, or D) and explain your reasoning."
        )
    
    def review_answer(self, question, initial_answer, initial_reasoning):
        """
        Review the initial answer and provide a critique and improved answer.
        
        Args:
            question: The original question
            initial_answer: The initial answer to review
            initial_reasoning: The reasoning behind the initial answer
            
        Returns:
            Dict containing:
            - answer: The reviewed answer (A, B, C, or D)
            - reasoning: The reasoning behind the reviewed answer
            - critique: A critique of the initial answer
        """
        prompt = (
            "Review the question and the initial answer thoroughly. "
            "Assess its strengths and weaknesses, and then determine provide an improved answer with better reasoning if applicable. "
            "Select the best final answer and provide your reasoning.\n"
            f"Original Question: {question}\n"
            f"Initial Answer: {initial_answer}\n"
            f"Initial Reasoning: {initial_reasoning}"
        )
        response = self.generate_response(prompt)
        return response