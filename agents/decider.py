'''
Initial Generator Agent -> Reviewer Agent -> Challenger Agent -> Refiner Agent -> (Decider Agent)
The decider agent aggregates candidate answers and their reasoning, and selects the optimal final answer
'''

import os
import json
from base_agent import Agent
from answer_schema import AnswerSchema

FUNCTION_SCHEMA = {
    "name": "decide_final_answer",
    "description": "Select the final best candidate answer based on critic reasoning and research evidence. Provide the answer and your rationale.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["A", "B", "C", "D"],
                "description": "The final selected answer choice."
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation combining critical reasoning and evidence supporting the final answer."
            }
        },
        "required": ["answer", "reasoning"]
    }
}

class ConsensusArbiterAgent(Agent):
    def __init__(self, topic, model="gpt-4o-mini", topic_roles_json="config/topic_roles.json"):
        self.role_description = "You are a senior financial QA evaluator, combining critical review and research evidence to decide the final answer."
        super().__init__(model=model, function_schema=FUNCTION_SCHEMA, pyd_model=None)
    
    def system_prompt(self):
        return (
            f"{self.role_description}\n"
            "Your job is to consolidate the critic's improved reasoning and the researcher's evidence-backed answer. "
            "Decide on the final answer and provide combined rationale."
        )
    
    def process(self, question, critic_answer, critic_reasoning, research_answer, research_evidence):
        prompt = (
            "Consolidate the following critical reasoning and research evidence to choose the final answer.\n"
            f"Original Question: {question}\n\n"
            f"Critic's Answer: {critic_answer}\n"
            f"Critic Reasoning: {critic_reasoning}\n\n"
            f"Researcher's Answer: {research_answer}\n"
            f"Researcher's Evidence: {research_evidence}\n\n"
            "Provide the final answer choice and combined explanation."
        )
        return self.generate_response(prompt) 
