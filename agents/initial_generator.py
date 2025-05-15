import os
import json
from base_agent import Agent
from answer_schema import AnswerSchema
    
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
            }
        },
        "required": ["answer", "reasoning"]
    }
}

class InitialGeneratorAgent(Agent):
    def __init__(self, model = "gpt-4o-mini"): 
        super().__init__(model=model, function_schema=FUNCTION_SCHEMA, pyd_model=AnswerSchema)
    
    def system_prompt(self):
        return (
            "You are a financial analyst AI. Think step by step and propose an answer to the question, explaining your reasoning. "
            "Do not use any external sources or references. "
            "Select one letter: A, B, C, or D."
        )
    
    def process(self, question):
        ''' returns dict. format 
        {
            "answer": "A",
            "reasoning": "Because option A best fits the data."
        }
        '''
        prompt = ( 
            "Answer the following question and provide your reasoning."           
            f"Question: {question}"
        )
        response = self.generate_response(prompt)
        return response 
