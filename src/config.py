"""
Configuration module for Open RAG.

This module centralizes all configuration settings including model parameters,
file paths, and default values used across the project.
"""

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
DOCUMENTS_DIR: Final[Path] = PROJECT_ROOT / "documents"


# =============================================================================
# Model Configuration
# =============================================================================

# Default Hugging Face model for text generation
# Options: "google/gemma-2b", "google/gemma-2b-it", "meta-llama/Llama-2-7b-hf", etc.
DEFAULT_LLM_MODEL: Final[str] = "google/gemma-2b-it"

# Sentence Transformer model for FAISS embeddings
SENTENCE_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"


# =============================================================================
# Generation Parameters
# =============================================================================

MAX_NEW_TOKENS: Final[int] = 512
TEMPERATURE: Final[float] = 0.6
TOP_P: Final[float] = 0.95
DO_SAMPLE: Final[bool] = True
REPETITION_PENALTY: Final[float] = 1.2


# =============================================================================
# Retrieval Configuration
# =============================================================================

# Chunking settings
CHUNK_SIZE: Final[int] = 1000
CHUNK_OVERLAP: Final[int] = 100

# FAISS settings
FAISS_TOP_K: Final[int] = 5

# BM25 settings
BM25_TOP_K: Final[int] = 5


# =============================================================================
# Hugging Face Configuration
# =============================================================================

def get_hf_token() -> str | None:
    """
    Retrieve the Hugging Face token from environment variables.
    
    Returns:
        str or None: The HF token if set, None otherwise.
    """
    return os.environ.get("HF_TOKEN")


# =============================================================================
# OpenAI Configuration
# =============================================================================

def get_openai_api_key() -> str | None:
    """
    Retrieve the OpenAI API key from environment variables.
    
    Returns:
        str or None: The OpenAI API key if set, None otherwise.
    """
    return os.environ.get("OPENAI_API_KEY")
