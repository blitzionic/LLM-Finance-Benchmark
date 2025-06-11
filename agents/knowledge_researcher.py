import os
import json
from .base_agent import Agent
from .pyd_schema import AnswerSchema
from typing import List
import re

class KnowledgeResearcherAgent(Agent):
    def __init__(self, model, provider, api_key=None, query_engine=None):
        self.query_engine = query_engine
        self.role_description = "You are a research assistant with access to financial documents and external data sources."
        # Initialize without function schema since we want direct text responses
        super().__init__(model=model, provider=provider, function_schema=None, pyd_model=None, api_key=api_key)

    def system_prompt(self):
        return (
            f"{self.role_description}\n"
            "Your task is to summarize evidence into clear, concise bullet points. "
            "Focus on extracting the most relevant information that could help answer the question. "
            "Format your response as a list of bullet points, starting each point with a dash (-)."
        )

    def retrieve_evidence(self, query, top_k=3):
        try:
            if self.query_engine:
                response = self.query_engine.query(query)
                # remove pdf markers from retrieved chunks
                retrieved_docs = [
                    node.node.text.replace("Final PDF to printer", "").replace("Page", "").replace("page", "").strip() 
                    for node in response.source_nodes
                ]
                print(f"Retrieved docs: {retrieved_docs}")
                return retrieved_docs
            else:
                return ["No query engine available."]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def summarize_evidence(self, evidence_chunks: List[str]) -> str:
        """Summarize the retrieved evidence into bullet points"""
        # Clean and join the evidence chunks
        cleaned_chunks = []
        for chunk in evidence_chunks:
            # Remove page numbers and chapter numbers
            chunk = re.sub(r'\d+\s+Chapter\s+\d+', '', chunk)
            # Remove standalone numbers
            chunk = re.sub(r'^\d+$', '', chunk)
            # Remove special characters and standardize dashes
            chunk = chunk.replace('–', '-').replace('●', '-')
            # Replace newlines with spaces
            chunk = chunk.replace('\n', ' ')
            # Remove extra whitespace
            chunk = ' '.join(chunk.split())
            if chunk.strip():  # Only add non-empty chunks
                cleaned_chunks.append(chunk)
        
        # Join cleaned chunks with proper spacing
        context = ' '.join(cleaned_chunks)
        
        # Create summary prompt
        summary_prompt = f"""Based on the following evidence, create a clear and concise summary in bullet points. Focus on the most relevant information and remove any duplicates or irrelevant details.
        Evidence:
        {context}
        Summary:"""
        
        # Generate summary
        summary_response = self.generate_response(summary_prompt)
        
        # Ensure we get a string response
        if isinstance(summary_response, dict):
            if "answer" in summary_response:
                summary = summary_response["answer"]
            else:
                summary = str(summary_response)
        else:
            summary = str(summary_response)
            
        print(f"Summarized evidence: {summary}")
        return summary

    def process_retrieval(self, question):
        """Process the retrieval and return the evidence"""
        # Get relevant context
        evidence_chunks = self.retrieve_evidence(question)
        
        # Clean the evidence chunks
        # lowercase all comments 
        cleaned_chunks = []
        for chunk in evidence_chunks:
            chunk = re.sub(r'\d+\s+Chapter\s+\d+', '', chunk)
            # Remove standalone numbers
            chunk = re.sub(r'^\d+$', '', chunk)
            # Remove special characters and standardize dashes
            chunk = chunk.replace('–', '-').replace('●', '-')
            # Replace newlines with spaces
            chunk = chunk.replace('\n', ' ')
            # Remove extra whitespace
            chunk = ' '.join(chunk.split())
            if chunk.strip():  # Only add non-empty chunks
                cleaned_chunks.append(chunk)
        
        # Join cleaned chunks with proper spacing
        cleaned_evidence = ' '.join(cleaned_chunks)
        
        print(f"Cleaned evidence: {cleaned_evidence}")
        return cleaned_evidence 