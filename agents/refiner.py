import os
import json
from base_agent import Agent
from answer_schema import AnswerSchema

class RefinerAgent(Agent):
    def __init__(self, topic, model="gpt-4o", topic_roles_json="topic_roles.json"):
        if os.path.exists(topic_roles_json):
            with open(topic_roles_json, 'r', encoding='utf-8') as f:
                roles = json.load(f)

        self.topic = topic
        self.role_desc = roles.get(topic, "You are a finance expert tasked with refining the current answer based"
                                            "on the feedback provided.")
        super().__init__(model=model, response_model=AnswerSchema)
    
    def system_prompt(self):
        return (
            f"{self.role_desc} "
            "You are a finance expert tasked with refining the current answer based on the feedback provided. "
            "Consider both the review and challenge feedback to integrate insights and produce the best final answer. "
            "Finally, select one letter as your answer: A, B, C, or D."
        )
    
    def process(self, current_answer: str, reviewer_feedback: str, challenger_feedback: str) -> str:
        prompt = (
            f"Current answer: '{current_answer}'\n"
            f"Reviewer feedback: '{reviewer_feedback}'\n"
            f"Challenger feedback: '{challenger_feedback}'\n"
            "Based on the above, please refine the answer and provide the best final answer. "
            "Select one letter: A, B, C, or D."
        )
        response = self.generate_response(prompt)
        # Assuming AnswerSchema defines an 'answer' field in the returned dictionary.
        return response.get("answer", "")
