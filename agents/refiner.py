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

class RefinerAgent(Agent):
    def __init__(self, topic, model="gpt-4o-mini", topic_roles_json="topic_roles.json"):
        if os.path.exists(topic_roles_json):
            with open(topic_roles_json, 'r', encoding='utf-8') as f:
                roles = json.load(f)
        else:
            roles = {}
            print(f"Warning: {topic_roles_json} not found. Using default role.")

        self.topic = topic
        self.role_description = roles.get(topic, "You are a finance expert tasked with refining the current answer based"
            "on the feedback provided.")
        super().__init__(model=model, function_schema=FUNCTION_SCHEMA, response_model=AnswerSchema)
    
    def system_prompt(self):
        return (
            f"{self.role_description} "
            "You are tasked with synthesizing multiple perspectives. Give the initial answer, the reviewer's answer, challenger's "
            "answer, produce a refined answer that combines the strongest elements from each response. Ensure that your final answer "
            " and reasoning are clear and coherent." 
            "Produce the best final answer. Select one letter as your answer: A, B, C, or D."
        )
    
    def process(self, question, initial_answer, initial_reasoning, reviewer_answer, revierwer_reasoning, challenger_answer,
                challenger_reasoning):
        prompt = (
            "Based on the following question and each question/reasoning, synthesize the strengths of each persecptive into a single "
            "refined answer and reasoning.\n"
            f"Original Question: {question}\n"
            f"Initial Answer: '{initial_answer}', Initial Reasoning: '{initial_reasoning}'\n"
            f"Reviewer Answer: '{reviewer_answer}, Reviewer Reasoning: '{revierwer_reasoning}'\n"
            f"Challenger Answer: '{challenger_answer}', Challenger Reasoning: '{challenger_reasoning}'\n"
        )
        response = self.generate_response(prompt)
        return response 
