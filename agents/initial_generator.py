import os
import json
from .base_agent import Agent
from .pyd_schema import AnswerSchema
    
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
                "description": "An explanation of the reasoning behind the chosen answer."
            }
        },
        "required": ["answer", "reasoning"]
    }
}

class InitialGeneratorAgent(Agent):
    def __init__(self, model, provider, api_key=None): 
        super().__init__(model=model, provider=provider, function_schema=FUNCTION_SCHEMA, pyd_model=AnswerSchema, api_key=api_key)
    
    def system_prompt(self):
        return (
            "Think step by step and propose an answer to the question, explaining your reasoning. "
            "If provided with evidence, use it to support your answer. "
            "If provided with feedback, carefully consider it to improve your answer. "
            "Select one letter: A, B, C, or D."
        )
    
    def process_question(self, question, context=None, feedback=None):
        """
        Process a question with optional context (RAG evidence) and feedback (from reviewer).
        
        Args:
            question: The question to answer
            context: Optional RAG evidence to support the answer
            feedback: Optional feedback from the reviewer to improve the answer
        """
        prompt = "Answer the following question and provide your reasoning."
        
        if context:
            prompt += f"\n\nUse the evidence to support your answer:\n{context}\n"
            
        if feedback:
            prompt += f"\n\nConsider the following feedback to improve your answer:\n{feedback}\n"
            
        prompt += f"\nQuestion: {question}"
        
        response = self.generate_response(prompt)
        return response 
