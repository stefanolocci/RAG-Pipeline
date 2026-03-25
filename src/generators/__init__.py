"""
Generators package for Open RAG.

Provides RAG generators that combine retrievers with LLMs:
- OpenRAG: Uses HuggingFace open-source models
- GPT_RAG: Uses OpenAI ChatGPT API
"""

try:
    from .open_rag import OpenRAG
    from .gpt_rag import GPT_RAG
except Exception:
    pass

__all__ = ["OpenRAG", "GPT_RAG"]
