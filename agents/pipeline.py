import os
from typing import Dict, List, Optional, Tuple
from base_agent import Agent
from .initial_generator import InitialGeneratorAgent
from .reviewer import CriticReviewerAgent
from .knowledge_researcher import KnowledgeResearcherAgent
from .decider import ConsensusArbiterAgent

class Pipeline:
    """
    A flexible pipeline that supports different configurations of agents for answering questions.
    Configurations:
    B-0: Base Generator -> Direct Answer
    B-1: RAG Retriever -> chunks + question -> Base Generator -> Answer
    B-2: Base Generator -> critic reviewer -> Answer
    B-3: RAG Retriever -> chunks + question -> Base Generator -> Critic Reviewer -> Answer
    B-4: RAG Retriever -> Chunks + question -> Base Generator -> Critic Reviewer Comments -> Arbiter
    """
    
    def __init__(
        self,
        config: str,
        provider: str = "openai",
        model: str = "gpt-4",
        query_engine = None,
        **kwargs
    ):
        """
        Initialize the pipeline with the specified configuration and LLM provider.
        
        Args:
            config: One of "B-0", "B-1", "B-2", "B-3", "B-4"
            provider: LLM provider to use ("openai", "runpod", "anthropic", "google")
            model: Model name/identifier
            query_engine: Optional RAG query engine for knowledge retrieval
            **kwargs: Additional provider-specific arguments
        """
        self.config = config
        self.provider = provider
        self.model = model
        self.query_engine = query_engine
        
        # Initialize agents based on configuration
        self.initial_generator = InitialGeneratorAgent(
            provider=provider,
            model=model,
            **kwargs
        )
        
        if config in ["B-2", "B-3", "B-4"]:
            self.critic_reviewer = CriticReviewerAgent(
                provider=provider,
                model=model,
                **kwargs
            )
            
        if config in ["B-1", "B-3", "B-4"]:
            if not query_engine:
                raise ValueError("Query engine required for configurations B-1, B-3, and B-4")
            self.knowledge_researcher = KnowledgeResearcherAgent(
                provider=provider,
                model=model,
                query_engine=query_engine,
                **kwargs
            )
            
        if config == "B-4":
            self.arbiter = ConsensusArbiterAgent(
                provider=provider,
                model=model,
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
            response = self.initial_generator.process_question(question)
            result.update({
                "answer": response.get("answer"),
                "reasoning": response.get("reasoning")
            })
            
        # B-1: RAG + Base Generator
        elif self.config == "B-1":
            evidence = self.knowledge_researcher.retrieve_evidence(question)
            response = self.initial_generator.process_question(
                question,
                context=evidence
            )
            result.update({
                "answer": response.get("answer"),
                "reasoning": response.get("reasoning"),
                "evidence": evidence
            })
            
        # B-2: Base Generator + Critic Review
        elif self.config == "B-2":
            initial_response = self.initial_generator.process_question(question)
            review = self.critic_reviewer.review_answer(
                question,
                initial_response.get("answer"),
                initial_response.get("reasoning")
            )
            result.update({
                "initial_answer": initial_response.get("answer"),
                "initial_reasoning": initial_response.get("reasoning"),
                "reviewed_answer": review.get("answer"),
                "reviewed_reasoning": review.get("reasoning"),
                "critique": review.get("critique")
            })
            
        # B-3: RAG + Base Generator + Critic Review
        elif self.config == "B-3":
            evidence = self.knowledge_researcher.retrieve_evidence(question)
            initial_response = self.initial_generator.process_question(
                question,
                context=evidence
            )
            review = self.critic_reviewer.review_answer(
                question,
                initial_response.get("answer"),
                initial_response.get("reasoning")
            )
            result.update({
                "initial_answer": initial_response.get("answer"),
                "initial_reasoning": initial_response.get("reasoning"),
                "reviewed_answer": review.get("answer"),
                "reviewed_reasoning": review.get("reasoning"),
                "critique": review.get("critique"),
                "evidence": evidence
            })
            
        # B-4: Full pipeline with Arbiter
        elif self.config == "B-4":
            evidence = self.knowledge_researcher.retrieve_evidence(question)
            initial_response = self.initial_generator.process_question(
                question,
                context=evidence
            )
            review = self.critic_reviewer.review_answer(
                question,
                initial_response.get("answer"),
                initial_response.get("reasoning")
            )
            final_decision = self.arbiter.make_decision(
                question,
                initial_response,
                review,
                evidence
            )
            result.update({
                "initial_answer": initial_response.get("answer"),
                "initial_reasoning": initial_response.get("reasoning"),
                "reviewed_answer": review.get("answer"),
                "reviewed_reasoning": review.get("reasoning"),
                "critique": review.get("critique"),
                "evidence": evidence,
                "final_answer": final_decision.get("answer"),
                "final_reasoning": final_decision.get("reasoning")
            })
            
        return result 