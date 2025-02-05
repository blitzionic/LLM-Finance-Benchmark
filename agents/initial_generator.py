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
            "feedback": {
                "type": "string",
                "description": "A brief explanation of the reasoning behind the chosen answer."
            }
        },
        "required": ["answer", "feedback"]
    }
}
class InitialGeneratorAgent(Agent):
    def __init__(self, model = "gpt-4o"):
        self.function_schema = FUNCTION_SCHEMA
        #super().__init(model = model)    
        super().__init__(model=model, pyd_model=AnswerSchema)
    
    def system_prompt(self):
        return (
            "Provide an answer to the following finance question(s)." 
            "Answer the following multiple-choice question by selecting one letter: A, B, C, or D."           
        )

    def process(self, question):
        model_response = self.generate_response(question)
        return model_response.get("answer", "")
