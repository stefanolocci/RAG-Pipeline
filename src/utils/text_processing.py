"""
Text processing utilities for Open RAG.

Provides functions for cleaning and processing text data.
"""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted characters and normalizing whitespace.
    
    Args:
        text: Input text to clean.
        
    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text.
        
    Returns:
        List of sentence strings.
    """
    if not text:
        return []
    
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text.
        chunk_size: Maximum characters per chunk.
        overlap: Overlapping characters between chunks.
        
    Returns:
        List of text chunks.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks
