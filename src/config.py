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
TEMPERATURE: Final[float] = 0.1
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


# =============================================================================
# SciFact / Gemini Configuration
# =============================================================================

# Gemini API key — set via environment variable or replace placeholder directly
GOOGLE_API_KEY: Final[str] = os.environ.get("GOOGLE_API_KEY", "")

# Gemini model names
EMBEDDING_MODEL: Final[str] = "gemini-embedding-001"
GENERATION_MODEL: Final[str] = "gemini-3.0-flash"

# Gemini embedding dimension
EMBEDDING_DIM: Final[int] = 384  # all-MiniLM-L6-v2 output dim

# Retrieval settings for SciFact
TOP_K: Final[int] = 5
BATCH_SIZE: Final[int] = 50

# Data paths (relative to project root)
CORPUS_PATH: Final[Path] = PROJECT_ROOT / "data" / "corpus.jsonl"
CLAIMS_DEV_PATH: Final[Path] = PROJECT_ROOT / "data" / "claims_dev.jsonl"
CLAIMS_TRAIN_PATH: Final[Path] = PROJECT_ROOT / "data" / "claims_train.jsonl"

# Index and output paths — default model (all-MiniLM-L6-v2)
FAISS_INDEX_PATH: Final[Path] = PROJECT_ROOT / "indices" / "scifact_faiss.index"
DOC_ID_MAP_PATH: Final[Path] = PROJECT_ROOT / "indices" / "doc_id_map.json"
PREDICTIONS_PATH: Final[Path] = PROJECT_ROOT / "outputs" / "predictions.jsonl"
DETAILED_LOG_PATH: Final[Path] = PROJECT_ROOT / "outputs" / "detailed_log.jsonl"

# =============================================================================
# Bio Model Configuration (Charangan/MedBERT)
# =============================================================================

BIO_EMBEDDING_MODEL: Final[str] = "Charangan/MedBERT"
BIO_EMBEDDING_DIM: Final[int] = 768  # BERT hidden size

# Index and output paths — bio model
BIO_FAISS_INDEX_PATH: Final[Path] = PROJECT_ROOT / "indices" / "scifact_faiss_biomodel.index"
BIO_DOC_ID_MAP_PATH: Final[Path] = PROJECT_ROOT / "indices" / "doc_id_map_biomodel.json"

# =============================================================================
# Cross-Encoder Reranking
# =============================================================================

CROSS_ENCODER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Number of FAISS candidates to fetch before reranking (must be >= TOP_K)
RERANKING_CANDIDATES_K: Final[int] = 20
