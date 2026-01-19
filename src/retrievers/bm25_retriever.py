"""
BM25-based retriever for Open RAG.

This module implements pure sparse term-based retrieval using BM25.

Example:
    >>> from open_RAG.src.retrievers import BM25Retriever
    >>> retriever = BM25Retriever(documents_dir="./documents", file_extension="pdf")
    >>> results = retriever.retrieve("What is prompt injection?")
"""

import logging
import re
from typing import Optional
from pathlib import Path

from rank_bm25 import BM25Okapi

from open_RAG.src.retrievers.base import BaseRetriever
from open_RAG.src.config import BM25_TOP_K


logger = logging.getLogger(__name__)


def simple_tokenize(text: str) -> list[str]:
    """Simple word tokenization - lowercase and split on non-alphanumeric."""
    return re.findall(r'\b\w+\b', text.lower())


class BM25Retriever(BaseRetriever):
    """
    Document retriever using BM25 for sparse term-based retrieval.
    
    Uses BM25 algorithm for keyword-based document ranking.
    
    Attributes:
        bm25: BM25Okapi instance for retrieval.
        tokenized_chunks: Tokenized document chunks.
    """
    
    def __init__(
        self,
        documents_dir: Optional[Path | str] = None,
        file_extension: str = "txt",
        top_k: int = BM25_TOP_K,
    ):
        """
        Initialize the BM25 retriever.
        
        Args:
            documents_dir: Directory containing documents.
            file_extension: File extension for documents (without dot).
            top_k: Default number of documents to retrieve.
        """
        # Initialize base class (loads documents)
        super().__init__(
            documents_dir=documents_dir,
            file_extension=file_extension,
            top_k=top_k,
        )
        
        # Build BM25 index
        self.bm25, self.tokenized_chunks = self._build_index()
    
    def _build_index(self) -> tuple[Optional[BM25Okapi], list[list[str]]]:
        """Build BM25 index from document chunks."""
        logger.info("Building BM25 index...")
        
        if not self.chunks:
            logger.warning("No documents to index")
            return None, []
        
        # Tokenize all chunks (extract content from tuples)
        tokenized = [simple_tokenize(content) for content, _ in self.chunks]
        
        bm25 = BM25Okapi(tokenized)
        
        logger.info(f"BM25 index built with {len(tokenized)} documents")
        return bm25, tokenized
    
    def retrieve(self, query: str, k: Optional[int] = None) -> list[tuple[str, str]]:
        """
        Retrieve relevant documents using BM25 ranking.
        
        Args:
            query: The query text.
            k: Number of documents to retrieve. Defaults to self.top_k.
            
        Returns:
            List of tuples (content, source_filename).
        """
        if not self.chunks or self.bm25 is None:
            return []
        
        k = k or self.top_k
        k = min(k, len(self.chunks))
        
        query_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        
        return [self.chunks[i] for i in top_indices]
