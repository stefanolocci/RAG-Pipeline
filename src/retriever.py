"""
SciFact FAISS retriever with optional bio-model embedding and cross-encoder reranking.

Usage:
    # Default model, no reranking
    docs = retrieve(claim)

    # Bio model (Charangan/MedBERT), no reranking
    docs = retrieve(claim, model="biomodel")

    # Default model + cross-encoder reranking
    docs = retrieve(claim, reranking=True)

    # Bio model + reranking
    docs = retrieve(claim, model="biomodel", reranking=True)

Each returned dict has keys: doc_id (int), text (str), score (float).
"""

import json
import logging

import faiss
import numpy as np
from sentence_transformers import CrossEncoder

from src.config import (
    CORPUS_PATH,
    FAISS_INDEX_PATH,
    DOC_ID_MAP_PATH,
    BIO_FAISS_INDEX_PATH,
    BIO_DOC_ID_MAP_PATH,
    SENTENCE_MODEL_NAME,
    BIO_EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL,
    TOP_K,
    RERANKING_CANDIDATES_K,
)
from src.gemini_client import embed_texts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model key helpers
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = {
    "default": {
        "index_path":   FAISS_INDEX_PATH,
        "map_path":     DOC_ID_MAP_PATH,
        "model_name":   SENTENCE_MODEL_NAME,
    },
    "biomodel": {
        "index_path":   BIO_FAISS_INDEX_PATH,
        "map_path":     BIO_DOC_ID_MAP_PATH,
        "model_name":   BIO_EMBEDDING_MODEL,
    },
}


# ---------------------------------------------------------------------------
# Module-level state (one slot per model key, loaded lazily)
# ---------------------------------------------------------------------------

_indices:   dict[str, faiss.IndexFlatIP] = {}
_doc_ids:   dict[str, list[int]]         = {}
_doc_texts: dict[int, str]               = {}   # shared — same corpus for all models
_cross_encoder: CrossEncoder | None      = None


def _load_resources(model: str) -> None:
    """Load FAISS index and doc_id map for the given model key (once)."""
    global _doc_texts

    if model in _indices:
        return

    cfg = _MODEL_CONFIGS[model]
    index_path = cfg["index_path"]
    map_path   = cfg["map_path"]

    logger.info(f"[{model}] Loading FAISS index from {index_path}")
    _indices[model] = faiss.read_index(str(index_path))

    with open(map_path, encoding="utf-8") as f:
        _doc_ids[model] = json.load(f)

    # Load corpus texts once (shared across models)
    if not _doc_texts:
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                _doc_texts[doc["doc_id"]] = doc["title"] + ". " + " ".join(doc["abstract"])

    logger.info(
        f"[{model}] Retriever ready — {_indices[model].ntotal} vectors, "
        f"{len(_doc_texts)} texts"
    )


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    claim: str,
    k: int = TOP_K,
    model: str = "default",
    reranking: bool = False,
) -> list[dict]:
    """
    Retrieve the top-k most relevant documents for a claim.

    Args:
        claim:     Biomedical claim string.
        k:         Number of documents to return.
        model:     "default" (all-MiniLM-L6-v2) or "biomodel" (MedBERT).
        reranking: If True, fetch RERANKING_CANDIDATES_K from FAISS then
                   rerank with a cross-encoder before returning top-k.

    Returns:
        List of dicts: [{"doc_id": int, "text": str, "score": float}, ...]
    """
    if model not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown model key '{model}'. Choose from: {list(_MODEL_CONFIGS)}")

    _load_resources(model)

    model_name = _MODEL_CONFIGS[model]["model_name"]
    candidates_k = RERANKING_CANDIDATES_K if reranking else k

    # --- FAISS retrieval ---
    query_vec = np.array(
        embed_texts([claim], task_type="RETRIEVAL_QUERY", model_name=model_name),
        dtype="float32",
    )
    faiss.normalize_L2(query_vec)

    scores, indices = _indices[model].search(query_vec, candidates_k)

    candidates = []
    for j, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        doc_id = _doc_ids[model][idx]
        candidates.append({
            "doc_id": doc_id,
            "text":   _doc_texts.get(doc_id, ""),
            "score":  float(scores[0][j]),
        })

    if not reranking:
        return candidates

    # --- Cross-encoder reranking ---
    logger.info(f"Reranking {len(candidates)} candidates with {CROSS_ENCODER_MODEL}...")
    ce = _get_cross_encoder()
    pairs = [(claim, doc["text"]) for doc in candidates]
    ce_scores = ce.predict(pairs)

    reranked = sorted(
        zip(ce_scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    return [
        {**doc, "score": float(ce_score)}
        for ce_score, doc in reranked[:k]
    ]
