"""
Base retriever module for Open RAG.

This module provides document loading, chunking utilities, and the base
retriever interface for FAISS and BM25 implementations.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from open_RAG.src.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Template
# =============================================================================

PROMPT_TEMPLATE = """You are an AI assistant designed to provide detailed, step-by-step responses.
Before providing an answer, you must first analyze the context provided and then provide a detailed response.

Context from retrieved documents:
{context}

User query:
{query}

Response:"""


# =============================================================================
# Document Loading Utilities
# =============================================================================

def load_documents_from_directory(
    directory: Path | str,
    file_extension: str = "txt"
) -> list[tuple[str, str]]:
    """
    Load all documents with specified extension from a directory.
    
    Supports text files (.txt, .md, etc.) and PDF files (.pdf).
    
    Args:
        directory: Path to the directory containing documents.
        file_extension: File extension to look for (without dot). Default: 'txt'
        
    Returns:
        List of tuples (content, filename).
    """
    docs = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory not found: {dir_path}")
        return docs
    
    # Ensure extension doesn't have a leading dot
    ext = file_extension.lstrip('.').lower()
    files = list(dir_path.glob(f"*.{ext}"))
    logger.info(f"Found {len(files)} .{ext} files in {dir_path}")
    
    for filepath in files:
        try:
            if ext == "pdf":
                content = _read_pdf(filepath)
            else:
                content = filepath.read_text(encoding='utf-8')
            
            content = content.strip()
            if content:
                docs.append((content, filepath.name))
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
    
    logger.info(f"Loaded {len(docs)} documents")
    return docs


def _read_pdf(filepath: Path) -> str:
    """Extract text content from a PDF file."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF support. "
            "Install it with: pip install pdfplumber"
        )
    
    text_content = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                # x_tolerance and y_tolerance help with detection of spaces and logic layout
                page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                
                if page_text:
                    # Clean up some common PDF extraction issues
                    # Replace multiple newlines with double newline (paragraph break)
                    # Join hyphenated words at line ends (e.g. "exam-\nple" -> "example")
                    # This is a basic heuristic
                    clean_text = page_text.replace('-\n', '')
                    text_content.append(clean_text)
    except Exception as e:
        logger.error(f"Failed to read PDF {filepath}: {e}")
        return ""
    
    return "\n\n".join(text_content)


def chunk_documents(
    documents: list[tuple[str, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[tuple[str, str]]:
    """
    Split documents into smaller chunks using recursive splitting.
    
    Tries to split by paragraphs, then sentences, then words to keep
    semantically related text together.
    
    Args:
        documents: List of tuples (content, filename).
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.
        
    Returns:
        List of tuples (chunk_content, source_filename).
    """
    chunks = []
    
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    for doc_content, filename in documents:
        doc_chunks = _recursive_split(doc_content, chunk_size, overlap, separators)
        for chunk in doc_chunks:
            chunks.append((chunk, filename))
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def _recursive_split(
    text: str, 
    chunk_size: int, 
    overlap: int, 
    separators: list[str]
) -> list[str]:
    """
    Recursively split text by the given separators.
    """
    final_chunks = []
    
    # Base case: text fits in chunk
    if len(text) <= chunk_size:
        return [text]
    
    # If no separators left, force split
    if not separators:
        # Just hard split by size
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    
    # Try current separator
    separator = separators[0]
    next_separators = separators[1:]
    
    # If separator is empty string, we are at character level
    if separator == "":
        return _recursive_split(text, chunk_size, overlap, [])
        
    splits = text.split(separator)
    
    # Determine the buffer (accumulated splits)
    current_chunk = []
    current_len = 0
    
    for split in splits:
        # If the split itself is too big, recurse on it
        if len(split) > chunk_size:
            # First, save what we have so far
            if current_chunk:
                joined = separator.join(current_chunk)
                final_chunks.append(joined)
                current_chunk = []
                current_len = 0
            
            # Recurse on the big split
            sub_chunks = _recursive_split(split, chunk_size, overlap, next_separators)
            final_chunks.extend(sub_chunks)
            continue
            
        # If adding this split exceeds chunk size, save current_chunk
        # Add separator length to calculation
        split_len = len(split) + (len(separator) if current_chunk else 0)
        
        if current_len + split_len > chunk_size:
            if current_chunk:
                joined = separator.join(current_chunk)
                final_chunks.append(joined)
                
                # Handling overlap is tricky in simple recursive split
                # Simplify: start new chunk with current split
                # Better overlap logic would keep some previous items
                
                # Simple overlap attempt: keep last item if it fits with new one
                # (Removing complicate overlap logic for stability in this iteration)
                current_chunk = [split]
                current_len = len(split)
            else:
                # Edge case where single split matches chunk size exactly
                final_chunks.append(split)
                current_chunk = []
                current_len = 0
        else:
            current_chunk.append(split)
            current_len += split_len
            
    # Add remaining
    if current_chunk:
        final_chunks.append(separator.join(current_chunk))
        
    return final_chunks


def create_prompt(query: str, context_docs: list[str]) -> str:
    """
    Create a prompt with retrieved context.
    
    Args:
        query: The user's query.
        context_docs: Retrieved relevant documents.
        
    Returns:
        Formatted prompt string.
    """
    context = "\n\n".join(context_docs) if context_docs else "No relevant context found."
    return PROMPT_TEMPLATE.format(context=context, query=query)


# =============================================================================
# Base Retriever Interface
# =============================================================================

class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers.
    
    Provides the interface for FAISS and BM25 retriever implementations.
    Retrievers are pure document retrieval - no LLM dependency.
    
    Attributes:
        documents: List of loaded documents (content, filename).
        chunks: Chunked documents for retrieval.
        top_k: Default number of documents to retrieve.
    """
    
    def __init__(
        self,
        documents_dir: Optional[Path | str] = None,
        file_extension: str = "txt",
        top_k: int = 5,
    ):
        """
        Initialize the base retriever.
        
        Args:
            documents_dir: Directory containing documents. Defaults to config.
            file_extension: File extension for documents (without dot).
            top_k: Default number of documents to retrieve.
        """
        self.top_k = top_k
        self.file_extension = file_extension
        
        # Load documents
        docs_dir = Path(documents_dir) if documents_dir else DOCUMENTS_DIR
        self.documents = load_documents_from_directory(docs_dir, file_extension)
        self.chunks = chunk_documents(self.documents)
    
    @abstractmethod
    def retrieve(self, query: str, k: Optional[int] = None) -> list[tuple[str, str]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The query text.
            k: Number of documents to retrieve. Defaults to self.top_k.
            
        Returns:
            List of tuples (content, source_filename).
        """
        pass
    
    def get_context(self, query: str, k: Optional[int] = None) -> dict:
        """
        Retrieve documents and format for RAG.
        
        Args:
            query: The query text.
            k: Number of documents to retrieve.
            
        Returns:
            Dictionary with 'documents', 'sources', and 'context_text'.
        """
        retrieved = self.retrieve(query, k)
        
        doc_contents = [content for content, _ in retrieved]
        sources = [filename for _, filename in retrieved]
        
        return {
            "documents": doc_contents,
            "sources": sources,
            "context_text": "\n\n".join(doc_contents) if doc_contents else "No relevant context found.",
        }
