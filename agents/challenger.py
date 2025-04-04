# agents/challenger.py

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
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

class ChallengerAgent(Agent):
    def __init__(self, topic, model="gpt-4o-mini", query_engine=None, topic_roles_json="topic_roles.json"):
        if os.path.exists(topic_roles_json):
            with open(topic_roles_json, 'r', encoding='utf-8') as f:
                roles = json.load(f)
        else:
            roles = {}
            print(f"Warning: {topic_roles_json} not found. Using default role.")
    
        self.topic = topic
        self.role_description = roles.get(topic, "You are a critical financial analyst, experienced in scrutinizing financial data.")
        self.query_engine = query_engine
        super().__init__(model=model, function_schema=FUNCTION_SCHEMA, response_model=AnswerSchema)
    
    def system_prompt(self):
        return (
            f"{self.role_description} "
            "Using any relevant context, critically evaluate the current answer and determine what you believe is the correct answer. "
            "Answer by selecting one letter: A, B, C, or D."
        )
    
    def retrieve_relevant_docs(self, query, top_k=5):
        try:
            # returns most relevent chunks from financial source
            response = self.query_engine.query(query)
            retrieved_docs = [node.node.txt for node in response.source_nodes]
            return retrieved_docs
        except Exception as e:
            print(f"Error in retrieving relevant documents: {e}")
            return []
    
    def process(self, question, previous_model_answer):
        # retrieve relevant context documents based on the question.
        retrieved_docs = self.retrieve_relevant_docs(question)
        context = "\n\n".join(retrieved_docs)

        # build a prompt that includes the current answer, the question, and the retrieved context
        prompt = (
            f"Current answer: '{previous_model_answer}'.\n"
            f"Question: {question}\n"
            f"Relevant context:\n{context}\n\n"
            "Based on the above, identify any potential flaws or overlooked aspects in the current answer. "
            "Then, determine what you believe is the correct answer by selecting one letter: A, B, C, or D."
        )
        response = self.generate_response(prompt)
        return response.get("answer", "")
