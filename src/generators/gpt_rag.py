"""
GPT_RAG - RAG Generator using OpenAI ChatGPT API.

This module provides RAG generation using OpenAI's models (gpt-4, gpt-4o-mini, etc.).

Example:
    >>> from open_RAG.src.retrievers import FAISSRetriever
    >>> from open_RAG.src.generators import GPT_RAG
    >>> retriever = FAISSRetriever(documents_dir="./documents", file_extension="pdf")
    >>> rag = GPT_RAG(retriever, model="gpt-4o-mini")
    >>> result = rag.query("What is prompt injection?")
"""

import logging
from typing import Optional

from openai import OpenAI

from open_RAG.src.retrievers.base import BaseRetriever, create_prompt
from open_RAG.src.config import get_openai_api_key


logger = logging.getLogger(__name__)


class GPT_RAG:
    """
    RAG Generator using OpenAI ChatGPT API.
    
    Combines a retriever with OpenAI's API for RAG-based generation.
    
    Attributes:
        retriever: Document retriever instance (FAISS or BM25).
        client: OpenAI client.
        model: OpenAI model name.
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Initialize GPT_RAG.
        
        Args:
            retriever: A retriever instance (FAISSRetriever or BM25Retriever).
            model: OpenAI model name (gpt-4, gpt-4o-mini, gpt-3.5-turbo).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in your .env file."
            )
        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized with model: {model}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt: The input prompt with context.
            
        Returns:
            Generated text response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, question: str, k: Optional[int] = None) -> dict:
        """
        Perform RAG: retrieve relevant docs and generate response.
        
        Args:
            question: The user's question.
            k: Number of documents to retrieve.
            
        Returns:
            Dictionary with 'response', 'retrieved_docs', 'sources', and 'prompt'.
        """
        # Retrieve context
        context = self.retriever.get_context(question, k)
        
        # Create prompt
        prompt = create_prompt(question, context["documents"])
        
        # Generate response
        response = self.generate(prompt)
        
        return {
            "response": response,
            "retrieved_docs": context["documents"],
            "sources": context["sources"],
            "prompt": prompt,
        }
