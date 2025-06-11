import os
from typing import Dict, List, Optional, Tuple
from .base_agent import Agent
from .initial_generator import InitialGeneratorAgent
from .reviewer import CriticReviewerAgent
from .knowledge_researcher import KnowledgeResearcherAgent

class Pipeline:
    """
    A flexible pipeline that supports different configurations of agents for answering questions.
    Configurations:
    B-0: Base Generator -> Direct Answer
    B-1: RAG Retriever -> chunks + question -> Base Generator -> Answer
        - top 3 chunks
        - no summarization of evidence 
        - streamlined through B3
    B-2: Base Generator -> critic reviewer -> Feedback -> Base Generator -> Answer
    B-3: RAG Retriever -> chunks + question -> Base Generator -> Critic Reviewer -> Feedback -> Base Generator -> Answer
        - Also provides B-1 equivalent results through initial_answer and initial_reasoning
    """
    
    def __init__(
        self,
        config: str,
        provider: str,
        model: str,
        query_engine = None,
        **kwargs
    ):
        """
        Initialize the pipeline with the specified configuration and LLM provider.
        
        Args: 
            config: One of "B-0", "B-2", "B-3" 
            provider: LLM provider to use ("openai", "runpod", "anthropic", "google")
            model: Model name/identifier
            query_engine: Optional RAG query engine for knowledge retrieval
            **kwargs: Additional provider-specific arguments
        """
        if config not in ["B-0", "B-2", "B-3"]:
            raise ValueError("Config must be one of: B-0, B-2, B-3")
            
        self.config = config
        self.provider = provider
        self.model = model
        self.query_engine = query_engine
        
        # init agents based on configuration 
        self.initial_generator = InitialGeneratorAgent(
            provider=provider,
            model=model,
            **kwargs
        )
        # init critic reviewer
        if config in ["B-2", "B-3"]:
            self.critic_reviewer = CriticReviewerAgent(
                provider=provider,
                model=model,
                **kwargs
            )
        # init knowledge researcher (RAG)
        if config == "B-3":
            if not query_engine:
                raise ValueError("Query engine required for configuration B-3")
            self.knowledge_researcher = KnowledgeResearcherAgent(
                provider=provider,
                model=model,
                query_engine=query_engine,
                **kwargs
            )
    
    def process(self, question: str) -> Dict:
        """
        Process a question through the pipeline based on the configuration.
        
        Args:
            question: The question to process
            
        Returns:
            Dict containing the results
        """
    
        result = {
            "question": question,
            "config": self.config
        }
        
        # B-0: Direct answer from Base Generator
        if self.config == "B-0":
            initial_response = self.initial_generator.process_question(question)
            result.update({
                "answer": initial_response.get("answer"),
                "reasoning": initial_response.get("reasoning")
            })
            
        # B-2: Base Generator -> answer -> critic reviewer -> Feedback -> Base Generator -> Answer
        elif self.config == "B-2":
            initial_response = self.initial_generator.process_question(question)
            review = self.critic_reviewer.review_answer(question, initial_response.get("answer"), initial_response.get("reasoning"))
            # feed back to initial generator
            print("Review: ", review)
            improved_response = self.initial_generator.process_question(question, context=review.get("critique"))
            result.update({
                "initial_answer": initial_response.get("answer"),
                "initial_reasoning": initial_response.get("reasoning"),
                "critique": review,
                "improved_answer": improved_response.get("answer"),
                "improved_reasoning": improved_response.get("reasoning")
            })
            
        # B-3: RAG Retriever -> chunks + question -> Base Generator -> Critic Reviewer -> Feedback -> Base Generator -> Answer
        elif self.config == "B-3":
            evidence = self.knowledge_researcher.process_retrieval(question)
            initial_response = self.initial_generator.process_question(question, context=evidence)
            result.update({
                "evidence": evidence,
                "initial_answer": initial_response.get("answer"),
                "initial_reasoning": initial_response.get("reasoning")
            })
            review = self.critic_reviewer.review_answer(question, initial_response.get("answer"), initial_response.get("reasoning"))
            improved_response = self.initial_generator.process_question(question, context=review.get("critique"))
            result.update({
                "critique": review,
                "improved_answer": improved_response.get("answer"),
                "improved_reasoning": improved_response.get("reasoning")
            })
            
        return result 