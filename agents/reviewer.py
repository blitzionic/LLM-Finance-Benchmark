import os
import json
from base_agent import Agent
from answer_schema import AnswerSchema

# Intitial Generator -> Reviewer -> Challenger Agent 
# The Reviewer Agent is intended to “review” the initial generated solution. 
# It reviews the solution returns its own answer. 

class ReviewerAgent(Agent):
    def __init__(self, topic, model="gpt-4o", topic_roles_json = "topic_roles.json"):
        # Use the Pydantic model for structured output.
        if os.path.exists(topic_roles_json):
          with open(topic_roles_json, 'r', encoding='utf-8') as f:
            roles = json.load(f) 
            
        self.topic = topic
        self.role_desc = topic.get(topic,"")
        super().__init__(model=model, response_model=AnswerSchema)
    
    def system_prompt(self):
        return (
            f"{self.role_description} "
            "As a financial expert, you are tasked with evaluating the answer's strengths and weeknesses. "
            "Review the provided answer and determine what you believe is the correct answer for the multiple-choice finance question. "
            "Finally, answer by selecting one letter you beleive is correct: A, B, C, or D."
        )
    
    def process(self, current_answer):
        prompt = (
            f"Given the current answer '{current_answer}', what is the correct answer to the multiple-choice question? "
            "Select one letter: A, B, C, or D."
        )
        response = self.generate_response(prompt)
        return response.get("answer", "")
