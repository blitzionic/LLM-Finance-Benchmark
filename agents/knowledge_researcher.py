import os
import json
from base_agent import Agent

FUNCTION_SCHEMA = {
    "name": "research_answer",
    "description": "Retrieve evidence from external sources and provide a candidate answer. The answer must be one letter among A, B, C, or D, and include a list of evidence citations.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["A", "B", "C", "D"],
                "description": "The candidate answer."
            },
            "evidence": {
                "type": "string",
                "description": "Evidence supporting your answer, including citations from reliable sources."
            }
        },
        "required": ["answer", "evidence"]
    }
}

class KnowledgeResearcherAgent(Agent):
    def __init__(self, topic="finance", model="gpt-4o-mini", query_engine=None):
        self.topic = topic
        self.query_engine = query_engine
        self.role_description = "You are a research assistant with access to financial documents and external data sources."
        super().__init__(model=model, function_schema=FUNCTION_SCHEMA, pyd_model=None)

    def system_prompt(self):
        return (
            f"{self.role_description}\n"
            "Search for information that answers the question and use it to provide a supported answer. "
            "Analyze the evidence objectively and determine the most accurate answer based on facts. "
            "Select one letter: A, B, C, or D, and provide evidence from reliable sources."
        )

    def retrieve_relevant_docs(self, query, top_k=3):
        try:
            if self.query_engine:
                response = self.query_engine.query(query)
                retrieved_docs = [node.node.text for node in response.source_nodes]
                return retrieved_docs
            else:
                return ["No query engine available."]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def process(self, question):
        # Retrieve relevant context documents based on the question
        retrieved_docs = self.retrieve_relevant_docs(question)
        context = "\n\n".join(retrieved_docs)

        # Build a prompt that includes the retrieved context
        prompt = (
            "Based on the provided context and your financial knowledge, answer the following question. "
            "Analyze the evidence objectively and determine the correct answer.\n" 
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Provide your answer and supporting evidence."
        )
        response = self.generate_response(prompt)
        return response 