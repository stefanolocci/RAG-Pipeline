"""
FAISS-based retriever for Open RAG.

This module implements pure dense vector retrieval using FAISS.

Example:
    >>> from open_RAG.src.retrievers import FAISSRetriever
    >>> retriever = FAISSRetriever(documents_dir="./documents", file_extension="pdf")
    >>> results = retriever.retrieve("What is prompt injection?")
"""

import logging
from typing import Optional
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from open_RAG.src.retrievers.base import BaseRetriever
from open_RAG.src.config import SENTENCE_MODEL_NAME, FAISS_TOP_K


logger = logging.getLogger(__name__)


class FAISSRetriever(BaseRetriever):
    """
    Document retriever using FAISS for dense vector similarity search.
    
    Uses SentenceTransformers to encode documents into dense vectors and
    FAISS for efficient nearest neighbor search.
    
    Attributes:
        sentence_model: SentenceTransformer model for encoding.
        index: FAISS index for similarity search.
    """
    
    def __init__(
        self,
        documents_dir: Optional[Path | str] = None,
        file_extension: str = "txt",
        top_k: int = FAISS_TOP_K,
        sentence_model_name: str = SENTENCE_MODEL_NAME,
    ):
        """
        Initialize the FAISS retriever.
        
        Args:
            documents_dir: Directory containing documents.
            file_extension: File extension for documents (without dot).
            top_k: Default number of documents to retrieve.
            sentence_model_name: SentenceTransformer model for embeddings.
        """
        self.sentence_model_name = sentence_model_name
        
        # Load sentence transformer first
        logger.info(f"Loading SentenceTransformer: {sentence_model_name}")
        self.sentence_model = SentenceTransformer(sentence_model_name)
        logger.info("SentenceTransformer loaded")
        
        # Initialize base class (loads documents)
        super().__init__(
            documents_dir=documents_dir,
            file_extension=file_extension,
            top_k=top_k,
        )
        
        # Build FAISS index
        self.index = self._build_index()
    
    def _build_index(self) -> faiss.IndexFlatL2:
        """Build FAISS index from document chunks."""
        logger.info("Building FAISS index...")
        
        if not self.chunks:
            logger.warning("No documents to index")
            sample = self.sentence_model.encode(["sample"])
            dimension = sample.shape[1]
            return faiss.IndexFlatL2(dimension)
        
        # Extract content from tuples for encoding
        chunk_contents = [content for content, _ in self.chunks]
        
        embeddings = self.sentence_model.encode(
            chunk_contents,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {index.ntotal} vectors (dim={dimension})")
        return index
    
    def retrieve(self, query: str, k: Optional[int] = None) -> list[tuple[str, str]]:
        """
        Retrieve relevant documents using FAISS similarity search.
        
        Args:
            query: The query text.
            k: Number of documents to retrieve. Defaults to self.top_k.
            
        Returns:
            List of tuples (content, source_filename).
        """
        if not self.chunks:
            return []
        
        k = k or self.top_k
        k = min(k, len(self.chunks))
        
        query_vector = self.sentence_model.encode([query])
        _, indices = self.index.search(
            np.array(query_vector).astype('float32'), k
        )
        
        return [self.chunks[i] for i in indices[0] if i >= 0]
