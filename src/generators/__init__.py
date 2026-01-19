"""
Generators package for Open RAG.

Provides RAG generators that combine retrievers with LLMs:
- OpenRAG: Uses HuggingFace open-source models
- GPT_RAG: Uses OpenAI ChatGPT API
"""

from open_RAG.src.generators.open_rag import OpenRAG
from open_RAG.src.generators.gpt_rag import GPT_RAG

__all__ = ["OpenRAG", "GPT_RAG"]
