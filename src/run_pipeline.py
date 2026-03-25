"""
Main entry point: run the SciFact verification pipeline on all dev claims.

Usage:
    uv run src/run_pipeline.py                                       # default
    uv run src/run_pipeline.py --model biomodel                      # MedBERT embeddings
    uv run src/run_pipeline.py --reranking                           # cross-encoder reranking
    uv run src/run_pipeline.py --few-shot                            # 3 few-shot examples in prompt
    uv run src/run_pipeline.py --model biomodel --reranking --few-shot

Output files are named by configuration to avoid overwriting previous runs:
    outputs/predictions.jsonl                              (default)
    outputs/predictions_biomodel.jsonl                     (--model biomodel)
    outputs/predictions_reranking.jsonl                    (--reranking)
    outputs/predictions_few_shot.jsonl                     (--few-shot)
    outputs/predictions_biomodel_reranking_few_shot.jsonl  (all three)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import os

from tqdm import tqdm

from src.config import CLAIMS_DEV_PATH, PROJECT_ROOT
from src.claim_verifier import verify
from src.evaluate import normalize_gold_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _output_paths(model: str, reranking: bool, few_shot: bool) -> tuple[Path, Path]:
    """Return (predictions_path, detailed_log_path) for the given config."""
    suffix_parts = []
    if model != "default":
        suffix_parts.append(model)
    if reranking:
        suffix_parts.append("reranking")
    if few_shot:
        suffix_parts.append("few_shot")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    out_dir = PROJECT_ROOT / "outputs"
    return (
        out_dir / f"predictions{suffix}.jsonl",
        out_dir / f"detailed_log{suffix}.jsonl",
    )


def load_claims(path=CLAIMS_DEV_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SciFact verification pipeline")
    parser.add_argument(
        "--model",
        choices=["default", "biomodel"],
        default="default",
        help="Embedding model: 'default' (all-MiniLM-L6-v2) or 'biomodel' (MedBERT)",
    )
    parser.add_argument(
        "--reranking",
        action="store_true",
        help="Apply cross-encoder reranking after FAISS retrieval",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        dest="few_shot",
        help="Include 3 labelled few-shot examples in the LLM prompt",
    )
    args = parser.parse_args()

    predictions_path, detailed_log_path = _output_paths(args.model, args.reranking, args.few_shot)
    os.makedirs(predictions_path.parent, exist_ok=True)

    logger.info(f"Config: model={args.model}, reranking={args.reranking}, few_shot={args.few_shot}")
    logger.info(f"Predictions → {predictions_path}")
    logger.info(f"Detailed log → {detailed_log_path}")

    claims = load_claims()
    logger.info(f"Loaded {len(claims)} dev claims")

    # Resume support: skip successfully-processed claims; retry errors
    done_ids: set[int] = set()
    error_ids: set[int] = set()
    if detailed_log_path.exists():
        with open(detailed_log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "ERROR" in str(entry.get("explanation", "")):
                        error_ids.add(entry["id"])
                    else:
                        done_ids.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    # Strip error entries from both files so they get rewritten cleanly
    for path in (predictions_path, detailed_log_path):
        if path.exists() and error_ids:
            kept = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line).get("id") not in error_ids:
                        kept.append(line)
                except (json.JSONDecodeError, KeyError):
                    kept.append(line)
            path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    logger.info(f"Resuming — {len(done_ids)} done, {len(error_ids)} errors will be retried")

    pred_file = open(predictions_path, "a", encoding="utf-8")
    log_file = open(detailed_log_path, "a", encoding="utf-8")

    try:
        for claim in tqdm(claims, desc="Verifying claims"):
            claim_id = claim["id"]
            if claim_id in done_ids:
                continue

            gold_label = normalize_gold_label(claim)

            try:
                result = verify(claim["claim"], model=args.model, reranking=args.reranking, few_shot=args.few_shot)
            except Exception as e:
                logger.error(f"Error on claim {claim_id}: {e}")
                result = {
                    "label": "NOT_ENOUGH_INFO",
                    "evidence_doc_ids": [],
                    "explanation": f"ERROR: {e}",
                    "retrieved_docs": [],
                }

            # predictions.jsonl — minimal format for evaluate.py
            prediction = {
                "id": claim_id,
                "label": result["label"],
                "evidence_doc_ids": result.get("evidence_doc_ids", []),
                "retrieved_doc_ids": [d["doc_id"] for d in result.get("retrieved_docs", [])],
            }
            pred_file.write(json.dumps(prediction) + "\n")
            pred_file.flush()

            # detailed_log.jsonl — full info for error analysis
            log_entry = {
                "id": claim_id,
                "claim": claim["claim"],
                "gold_label": gold_label,
                "predicted_label": result["label"],
                "correct": result["label"] == gold_label,
                "retrieved_docs": [
                    {"doc_id": d["doc_id"], "score": d["score"]}
                    for d in result.get("retrieved_docs", [])
                ],
                "evidence_doc_ids": result.get("evidence_doc_ids", []),
                "explanation": result.get("explanation", ""),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

    finally:
        pred_file.close()
        log_file.close()

    logger.info(f"Pipeline complete. Predictions saved to {predictions_path}")
    logger.info(f"Detailed log saved to {detailed_log_path}")


if __name__ == "__main__":
    main()
