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

class RefinerAgent(Agent):
    def __init__(self, topic, model="gpt-4o-mini", topic_roles_json="topic_roles.json"):
        if os.path.exists(topic_roles_json):
            with open(topic_roles_json, 'r', encoding='utf-8') as f:
                roles = json.load(f)
        else:
            roles = {}
            print(f"Warning: {topic_roles_json} not found. Using default role.")

        self.topic = topic
        self.role_desc = roles.get(topic, "You are a finance expert tasked with refining the current answer based"
            "on the feedback provided.")
        super().__init__(model=model, function_schema=FUNCTION_SCHEMA, response_model=AnswerSchema)
    
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
