"""
Full claim verification pipeline.

Orchestrates retrieval + LLM generation for a single claim:

    result = verify(claim_text)
    # -> {"label": "SUPPORTS", "evidence_doc_ids": [...], "explanation": "..."}
"""

import json
import logging
import time

from src.retriever import retrieve
from src.prompts import build_verification_prompt
from src.gemini_client import generate
from src.config import TOP_K

logger = logging.getLogger(__name__)

# Delay between LLM calls to stay under 10 RPM free-tier limit
_GENERATION_SLEEP_SEC = 15  # 4 RPM to stay safely under the 5 RPM free-tier limit


def verify(
    claim: str,
    k: int = TOP_K,
    model: str = "default",
    reranking: bool = False,
    few_shot: bool = False,
) -> dict:
    """
    Verify a biomedical claim end-to-end.

    1. Retrieves top-k documents from the FAISS index.
    2. Builds the prompt and calls the Gemini generation model.
    3. Parses and returns the JSON verdict.

    Args:
        claim:     Biomedical claim text.
        k:         Number of evidence documents to retrieve.
        model:     Embedding model key: "default" or "biomodel".
        reranking: If True, apply cross-encoder reranking after FAISS.
        few_shot:  If True, include 3 labelled examples in the prompt.

    Returns:
        Dict with keys:
            label            — "SUPPORTS" | "REFUTES" | "NOT_ENOUGH_INFO"
            evidence_doc_ids — list of int doc_ids cited by the LLM
            explanation      — natural-language justification
            retrieved_docs   — raw retrieved document list (for logging)
    """
    retrieved_docs = retrieve(claim, k=k, model=model, reranking=reranking)
    prompt = build_verification_prompt(claim, retrieved_docs, few_shot=few_shot)

    raw = generate(prompt).strip()

    # Strip markdown fences if the model wraps output anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse error for claim: {claim[:80]}...")
        result = {
            "label": "NOT_ENOUGH_INFO",
            "evidence_doc_ids": [],
            "explanation": "PARSE_ERROR: " + raw,
        }

    result["retrieved_docs"] = retrieved_docs

    # Rate-limit pause (caller may override by not using this function directly)
    time.sleep(_GENERATION_SLEEP_SEC)

    return result
