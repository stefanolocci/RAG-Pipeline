"""
Retrievers package for Open RAG.

Provides pure document retrieval implementations:
- FAISSRetriever: Dense vector similarity search
- BM25Retriever: Sparse term-based retrieval
"""

# To avoid unnecessary dependencies (like faiss) being loaded when not needed,
# we do not import the retriever classes here.
# Users should import directly from the submodules.

__all__ = []
