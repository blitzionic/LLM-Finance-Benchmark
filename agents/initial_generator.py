from .base_agent import Agent
from .answer_schema import AnswerSchema
    
class InitialGeneratorAgent(Agent):
    def __init__(self, model = "gpt-4o"):
        super().__init(model = model, pyd_response_model = AnswerSchema)    
    
    def system_prompt(self):
        return (
            "Provide an answer to the following finance question(s)." 
            "Answer the following multiple-choice question by selecting one letter: A, B, C, or D."           
        )

    def process(self, question):
        model_response = self.generate_response(question)
        return model_response.get("answer", "")
