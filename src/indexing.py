"""
Corpus indexing script for SciFact.

Run once per embedding model to build and persist a FAISS index:

    python src/indexing.py                   # default: all-MiniLM-L6-v2
    python src/indexing.py --model biomodel  # Charangan/MedBERT (768 dim)

Outputs (default):
    indices/scifact_faiss.index
    indices/doc_id_map.json

Outputs (biomodel):
    indices/scifact_faiss_biomodel.index
    indices/doc_id_map_biomodel.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import os

import faiss
import numpy as np

from src.config import (
    CORPUS_PATH,
    FAISS_INDEX_PATH,
    DOC_ID_MAP_PATH,
    EMBEDDING_DIM,
    BIO_FAISS_INDEX_PATH,
    BIO_DOC_ID_MAP_PATH,
    BIO_EMBEDDING_DIM,
    SENTENCE_MODEL_NAME,
    BIO_EMBEDDING_MODEL,
)
from src.gemini_client import embed_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_corpus(corpus_path=CORPUS_PATH) -> tuple[list[int], list[str]]:
    """Load corpus.jsonl and return (doc_ids, texts)."""
    doc_ids: list[int] = []
    texts: list[str] = []

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            full_text = doc["title"] + ". " + " ".join(doc["abstract"])
            doc_ids.append(doc["doc_id"])
            texts.append(full_text)

    logger.info(f"Loaded {len(doc_ids)} documents from corpus")
    return doc_ids, texts


def build_index(
    doc_ids: list[int],
    texts: list[str],
    model_name: str,
    embedding_dim: int,
) -> faiss.IndexFlatIP:
    """Embed texts with the given model and build a cosine-similarity FAISS index."""
    logger.info(f"Generating embeddings with {model_name}...")
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", model_name=model_name)

    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)  # cosine sim = inner product after normalisation

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(vectors)
    logger.info(f"FAISS index built: {index.ntotal} vectors (dim={embedding_dim})")
    return index


def save_index(
    index: faiss.IndexFlatIP,
    doc_ids: list[int],
    index_path: Path,
    map_path: Path,
) -> None:
    """Persist the FAISS index and doc_id mapping to disk."""
    os.makedirs(index_path.parent, exist_ok=True)

    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path}")

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    logger.info(f"Doc ID map saved to {map_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SciFact FAISS index")
    parser.add_argument(
        "--model",
        choices=["default", "biomodel"],
        default="default",
        help="Embedding model: 'default' (all-MiniLM-L6-v2) or 'biomodel' (MedBERT)",
    )
    args = parser.parse_args()

    if args.model == "biomodel":
        model_name    = BIO_EMBEDDING_MODEL
        embedding_dim = BIO_EMBEDDING_DIM
        index_path    = BIO_FAISS_INDEX_PATH
        map_path      = BIO_DOC_ID_MAP_PATH
    else:
        model_name    = SENTENCE_MODEL_NAME
        embedding_dim = EMBEDDING_DIM
        index_path    = FAISS_INDEX_PATH
        map_path      = DOC_ID_MAP_PATH

    logger.info(f"Indexing with model='{args.model}' ({model_name})")
    doc_ids, texts = load_corpus()
    index = build_index(doc_ids, texts, model_name, embedding_dim)
    save_index(index, doc_ids, index_path, map_path)
    logger.info("Indexing complete.")


if __name__ == "__main__":
    main()
