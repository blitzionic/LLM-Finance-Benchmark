# agents/challenger.py

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from base_agent import Agent
from answer_schema import AnswerSchema

class ChallengerAgent(Agent):
    def __init__(self, topic, model="gpt-4", index=None, embedding_model=None, documents=None, topic_roles_json="topic_roles.json"):
        if os.path.exists(topic_roles_json):
            with open(topic_roles_json, 'r', encoding='utf-8') as f:
                roles = json.load(f)
        else:
            roles = {}
            print(f"Warning: {topic_roles_json} not found. Using default role.")
        
        self.topic = topic
        # Lookup the role description for the given topic; default if not found.
        self.role_description = roles.get(topic, "You are a critical financial analyst, experienced in scrutinizing financial data.")
        
        # Set retrieval parameters.
        self.index = index
        self.embedding_model = embedding_model
        self.documents = documents
        
        # Use the Pydantic model for structured output.
        super().__init__(model=model, response_model=AnswerSchema)
    
    def system_prompt(self):
        return (
            f"{self.role_description} "
            "Using any relevant context, critically evaluate the current answer and determine what you believe is the correct answer. "
            "Answer by selecting one letter: A, B, C, or D."
        )
    
    def retrieve_relevant_documents(self, query, top_k=5):
        try:
            # Convert the query into an embedding vector using SentenceTransformer.
            query_embedding = self.embedding_model.encode([query])
            # Use FAISS to search for the top_k most similar documents.
            distances, indices = self.index.search(np.array(query_embedding).astype("float32"), top_k)
            retrieved_docs = [self.documents[i] for i in indices[0]]
            return retrieved_docs
        except Exception as e:
            print(f"Error in retrieve_relevant_documents: {e}")
            return []
    
    def process(self, current_answer, question):
        # Retrieve relevant context documents based on the question.
        retrieved_docs = self.retrieve_relevant_documents(question)
        context = "\n\n".join(retrieved_docs) if retrieved_docs else "No additional context available."
        
        # Build a prompt that includes the current answer, the question, and the retrieved context.
        prompt = (
            f"Current answer: '{current_answer}'.\n"
            f"Question: {question}\n"
            f"Relevant context:\n{context}\n\n"
            "Based on the above, identify any potential flaws or overlooked aspects in the current answer. "
            "Then, determine what you believe is the correct answer by selecting one letter: A, B, C, or D."
        )
        response = self.generate_response(prompt)
        return response.get("answer", "")
