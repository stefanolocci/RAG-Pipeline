"""
Comprehensive evaluation for the SciFact misinformation detection pipeline.

Produces:
  outputs/evaluation/<config>/
    classification/
      confusion_matrix.png         — normalised + raw counts heatmap
      per_class_metrics.png        — Precision / Recall / F1 per class
      label_distribution.png       — gold vs predicted label counts
    retrieval/
      precision_recall_at_k.png    — P@k and R@k bars, split by gold label
      score_distribution.png       — top-1 retrieval score distribution per label
      hit_rate_by_label.png        — fraction of claims where ≥1 gold doc was retrieved
    error_analysis/
      misclassification_heatmap.png — where errors concentrate (gold → pred)
      score_vs_correctness.png      — retrieval score for correct vs wrong predictions
    report.txt                      — full numeric summary

Usage:
    uv run src/evaluate.py                              # default config
    uv run src/evaluate.py --model biomodel             # MedBERT embeddings
    uv run src/evaluate.py --reranking                  # cross-encoder reranking
    uv run src/evaluate.py --model biomodel --reranking # both
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.config import CLAIMS_DEV_PATH, PREDICTIONS_PATH, DETAILED_LOG_PATH, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _config_suffix(model: str, reranking: bool, few_shot: bool = False) -> str:
    """Return the filename suffix and subdir name for a given config."""
    parts = []
    if model == "biomodel":
        parts.append("biomodel")
    if reranking:
        parts.append("reranking")
    if few_shot:
        parts.append("few_shot")
    return "_".join(parts) if parts else ""


def _get_paths(model: str = "default", reranking: bool = False, few_shot: bool = False):
    """Return (predictions_path, log_path, eval_out_dir) for the given config."""
    suffix = _config_suffix(model, reranking, few_shot)
    outputs = PROJECT_ROOT / "outputs"
    if suffix:
        preds_path = outputs / f"predictions_{suffix}.jsonl"
        log_path   = outputs / f"detailed_log_{suffix}.jsonl"
        out_dir    = PROJECT_ROOT / "outputs" / "evaluation" / suffix
    else:
        preds_path = PREDICTIONS_PATH
        log_path   = DETAILED_LOG_PATH
        out_dir    = PROJECT_ROOT / "outputs" / "evaluation"
    return preds_path, log_path, out_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
LABEL_COLORS = {
    "SUPPORTS":       "#2ecc71",
    "REFUTES":        "#e74c3c",
    "NOT_ENOUGH_INFO":"#95a5a6",
}
PALETTE = [LABEL_COLORS[l] for l in LABELS]

GOLD_LABEL_MAP = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}

# (output dirs are now computed dynamically in evaluate() based on CLI args)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _set_style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# ---------------------------------------------------------------------------
# Data loading & normalisation
# ---------------------------------------------------------------------------

def _load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_gold_label(claim: dict) -> str:
    if not claim["evidence"]:
        return "NOT_ENOUGH_INFO"
    for _, ev_list in claim["evidence"].items():
        return GOLD_LABEL_MAP.get(ev_list[0]["label"], ev_list[0]["label"])
    return "NOT_ENOUGH_INFO"


def gold_doc_ids(claim: dict) -> set[int]:
    return {int(d) for d in claim["evidence"]}


def _build_records(
    predictions_path=PREDICTIONS_PATH,
    claims_path=CLAIMS_DEV_PATH,
    log_path=DETAILED_LOG_PATH,
) -> pd.DataFrame:
    """Merge predictions + gold + detailed log into one DataFrame."""
    preds   = {p["id"]: p  for p in _load_jsonl(predictions_path)}
    gold_cl = {c["id"]: c  for c in _load_jsonl(claims_path)}
    logs    = {l["id"]: l  for l in _load_jsonl(log_path)} if Path(log_path).exists() else {}

    rows = []
    for cid, claim in gold_cl.items():
        if cid not in preds:
            continue
        pred = preds[cid]
        log  = logs.get(cid, {})

        gold_label = normalize_gold_label(claim)
        pred_label = pred.get("label", "NOT_ENOUGH_INFO")

        # Skip claims with pipeline errors
        if "ERROR" in str(log.get("explanation", "")):
            continue

        gold_ids   = gold_doc_ids(claim)
        retr_ids   = [int(x) for x in pred.get("retrieved_doc_ids", [])]
        k          = len(retr_ids)
        hits       = sum(1 for d in retr_ids if d in gold_ids)

        # Top-1 retrieval score (from detailed log)
        ret_docs   = log.get("retrieved_docs", [])
        top1_score = ret_docs[0]["score"] if ret_docs else None

        rows.append({
            "id":           cid,
            "gold_label":   gold_label,
            "pred_label":   pred_label,
            "correct":      gold_label == pred_label,
            "has_evidence": bool(gold_ids),
            "prec_at_k":    hits / k          if k and gold_ids else None,
            "recall_at_k":  hits / len(gold_ids) if gold_ids else None,
            "top1_score":   top1_score,
            "any_hit":      hits > 0           if gold_ids else None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1 — Classification plots
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, out_dir: Path):
    cm_raw  = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    short = ["SUPPORTS", "REFUTES", "NEI"]

    for ax, data, fmt, title in zip(
        axes,
        [cm_norm, cm_raw],
        [".2f",   "d"],
        ["Normalised (row %)", "Raw counts"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=short, yticklabels=short,
            linewidths=0.5, linecolor="white",
            ax=ax, cbar=True,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Gold", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

    fig.suptitle("Confusion Matrix — SciFact Verification", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)


def _plot_per_class_metrics(y_true, y_pred, out_dir: Path):
    precs = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    recs  = recall_score   (y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    f1s   = f1_score       (y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    x     = np.arange(len(LABELS))
    width = 0.26
    short = ["SUPPORTS", "REFUTES", "NEI"]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, precs, width, label="Precision", color="#3498db", alpha=0.88)
    b2 = ax.bar(x,          recs,  width, label="Recall",    color="#e67e22", alpha=0.88)
    b3 = ax.bar(x + width, f1s,   width, label="F1",         color="#9b59b6", alpha=0.88)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics — Precision / Recall / F1", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5)

    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    acc      = accuracy_score(y_true, y_pred)
    ax.text(0.01, 0.97, f"Accuracy: {acc:.3f}  |  Macro F1: {macro_f1:.3f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_dir / "per_class_metrics.png")
    plt.close(fig)


def _plot_label_distribution(y_true, y_pred, out_dir: Path):
    gold_counts = {l: y_true.count(l) for l in LABELS}
    pred_counts = {l: y_pred.count(l) for l in LABELS}
    short = ["SUPPORTS", "REFUTES", "NEI"]

    x     = np.arange(len(LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - width / 2, [gold_counts[l] for l in LABELS],
                width, label="Gold", color=PALETTE, alpha=0.9, edgecolor="white")
    b2 = ax.bar(x + width / 2, [pred_counts[l] for l in LABELS],
                width, label="Predicted", color=PALETTE, alpha=0.45,
                edgecolor=[LABEL_COLORS[l] for l in LABELS], linewidth=1.5)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=12)
    ax.set_ylabel("Number of claims", fontsize=12)
    ax.set_title("Label Distribution — Gold vs Predicted", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_dir / "label_distribution.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2 — Retrieval plots
# ---------------------------------------------------------------------------

def _plot_precision_recall_at_k(df: pd.DataFrame, out_dir: Path):
    ev_df = df[df["has_evidence"] & df["prec_at_k"].notna()].copy()
    if ev_df.empty:
        return

    summary = (
        ev_df.groupby("gold_label")[["prec_at_k", "recall_at_k"]]
        .mean()
        .reindex(["SUPPORTS", "REFUTES"])
    )
    overall = ev_df[["prec_at_k", "recall_at_k"]].mean()

    labels_plot = list(summary.index) + ["OVERALL"]
    precs  = list(summary["prec_at_k"])  + [overall["prec_at_k"]]
    recs   = list(summary["recall_at_k"]) + [overall["recall_at_k"]]

    x     = np.arange(len(labels_plot))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - width / 2, precs, width, label="Precision@k",
                color="#2980b9", alpha=0.88)
    b2 = ax.bar(x + width / 2, recs,  width, label="Recall@k",
                color="#27ae60", alpha=0.88)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Retrieval Precision@k and Recall@k", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_dir / "precision_recall_at_k.png")
    plt.close(fig)


def _plot_score_distribution(df: pd.DataFrame, out_dir: Path):
    score_df = df[df["top1_score"].notna()].copy()
    if score_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for label in LABELS:
        sub = score_df[score_df["gold_label"] == label]["top1_score"]
        if not sub.empty:
            sub.plot.kde(ax=ax, color=LABEL_COLORS[label], linewidth=2, label=label)

    ax.set_xlabel("Top-1 Retrieval Score (cosine similarity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Top-1 Retrieval Score Distribution by Gold Label", fontsize=14, fontweight="bold")
    ax.legend(title="Gold label", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_dir / "score_distribution.png")
    plt.close(fig)


def _plot_hit_rate_by_label(df: pd.DataFrame, out_dir: Path):
    ev_df = df[df["has_evidence"] & df["any_hit"].notna()].copy()
    if ev_df.empty:
        return

    summary = (
        ev_df.groupby("gold_label")["any_hit"]
        .agg(["mean", "count"])
        .reindex(["SUPPORTS", "REFUTES"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        summary["gold_label"],
        summary["mean"],
        color=[LABEL_COLORS[l] for l in summary["gold_label"]],
        width=0.45, alpha=0.88,
    )
    for bar, (_, row) in zip(bars, summary.iterrows()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.1%}\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Hit Rate (≥1 gold doc retrieved)", fontsize=12)
    ax.set_title("Retrieval Hit Rate by Gold Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "hit_rate_by_label.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3 — Error analysis plots
# ---------------------------------------------------------------------------

def _plot_misclassification_heatmap(y_true, y_pred, out_dir: Path):
    errors_gold, errors_pred = [], []
    for g, p in zip(y_true, y_pred):
        if g != p:
            errors_gold.append(g)
            errors_pred.append(p)

    if not errors_gold:
        return

    short_map = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES", "NOT_ENOUGH_INFO": "NEI"}
    cm = confusion_matrix(
        [short_map[l] for l in errors_gold],
        [short_map[l] for l in errors_pred],
        labels=["SUPPORTS", "REFUTES", "NEI"],
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=["SUPPORTS", "REFUTES", "NEI"],
        yticklabels=["SUPPORTS", "REFUTES", "NEI"],
        linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Predicted (errors only)", fontsize=12)
    ax.set_ylabel("Gold (errors only)", fontsize=12)
    ax.set_title(
        f"Misclassification Heatmap\n({len(errors_gold)} errors out of {len(y_true)} claims)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "misclassification_heatmap.png")
    plt.close(fig)


def _plot_score_vs_correctness(df: pd.DataFrame, out_dir: Path):
    score_df = df[df["top1_score"].notna()].copy()
    if score_df.empty:
        return

    score_df["outcome"] = score_df.apply(
        lambda r: f"{r['gold_label']}\n✓ correct" if r["correct"]
                  else f"{r['gold_label']}\n✗ wrong",
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Box plot
    order = []
    for l in LABELS:
        for suffix in ["✓ correct", "✗ wrong"]:
            key = f"{l}\n{suffix}"
            if key in score_df["outcome"].values:
                order.append(key)

    palette = {}
    for l in LABELS:
        palette[f"{l}\n✓ correct"] = LABEL_COLORS[l]
        palette[f"{l}\n✗ wrong"]   = "#bdc3c7"

    sns.boxplot(
        data=score_df, x="outcome", y="top1_score",
        order=[o for o in order if o in score_df["outcome"].values],
        palette=palette, ax=axes[0], linewidth=1.2,
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Top-1 Cosine Similarity", fontsize=11)
    axes[0].set_title("Retrieval Score by Outcome", fontsize=13, fontweight="bold")
    axes[0].tick_params(axis="x", labelsize=8)

    # KDE: correct vs incorrect
    for correct, label, color, ls in [
        (True,  "Correct",   "#27ae60", "-"),
        (False, "Wrong",     "#c0392b", "--"),
    ]:
        sub = score_df[score_df["correct"] == correct]["top1_score"]
        if not sub.empty:
            sub.plot.kde(ax=axes[1], color=color, linestyle=ls, linewidth=2, label=label)

    axes[1].set_xlabel("Top-1 Cosine Similarity", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Score Distribution: Correct vs Wrong", fontsize=13, fontweight="bold")
    axes[1].legend(framealpha=0.9)

    plt.suptitle("Retrieval Score vs Classification Correctness",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "score_vs_correctness.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4 — Text report
# ---------------------------------------------------------------------------

def _write_report(df: pd.DataFrame, y_true, y_pred, out_dir: Path, log_path=None):
    ev_df = df[df["has_evidence"] & df["prec_at_k"].notna()]
    acc      = accuracy_score(y_true, y_pred)
    mac_f1   = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    avg_prec = ev_df["prec_at_k"].mean()   if not ev_df.empty else float("nan")
    avg_rec  = ev_df["recall_at_k"].mean() if not ev_df.empty else float("nan")
    hit_rate = df[df["has_evidence"] & df["any_hit"].notna()]["any_hit"].mean()

    errors_excluded = 0
    if log_path and Path(log_path).exists():
        errors_excluded = sum(
            1 for r in _load_jsonl(log_path) if "ERROR" in str(r.get("explanation", ""))
        )

    lines = [
        "=" * 70,
        "  SciFact Misinformation Detection — Evaluation Report",
        "=" * 70,
        "",
        f"  Claims evaluated : {len(y_true)}",
        f"  Errors excluded  : {errors_excluded}",
        "",
        "── Classification ──────────────────────────────────────────────────",
        f"  Accuracy         : {acc:.4f}",
        f"  Macro F1         : {mac_f1:.4f}",
        "",
        "  Per-class breakdown:",
        classification_report(y_true, y_pred, labels=LABELS, zero_division=0),
        "── Retrieval ────────────────────────────────────────────────────────",
        f"  Avg Precision@k  : {avg_prec:.4f}",
        f"  Avg Recall@k     : {avg_rec:.4f}",
        f"  Hit Rate (≥1 gold doc retrieved): {hit_rate:.4f}",
        "",
        "── Error summary ────────────────────────────────────────────────────",
    ]

    total_errors = sum(1 for g, p in zip(y_true, y_pred) if g != p)
    lines.append(f"  Total errors     : {total_errors} / {len(y_true)} ({total_errors/len(y_true):.1%})")

    error_pairs = defaultdict(int)
    for g, p in zip(y_true, y_pred):
        if g != p:
            error_pairs[(g, p)] += 1
    for (g, p), cnt in sorted(error_pairs.items(), key=lambda x: -x[1]):
        lines.append(f"    {g:20s} → {p:20s}  ({cnt}x)")

    lines += ["", "=" * 70]

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    logger.info(f"Report written to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(model: str = "default", reranking: bool = False, few_shot: bool = False):
    predictions_path, log_path, out_base = _get_paths(model, reranking, few_shot)
    claims_path = CLAIMS_DEV_PATH

    out_cls = out_base / "classification"
    out_ret = out_base / "retrieval"
    out_err = out_base / "error_analysis"

    _set_style()

    for d in (out_cls, out_ret, out_err):
        d.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        logger.error(f"Predictions file not found: {predictions_path}")
        logger.error("Run the pipeline first with the matching --model / --reranking flags.")
        return

    logger.info(f"Evaluating config: model={model}, reranking={reranking}, few_shot={few_shot}")
    logger.info(f"Predictions : {predictions_path}")
    logger.info(f"Log         : {log_path}")
    logger.info(f"Output dir  : {out_base}")

    logger.info("Building evaluation records...")
    df = _build_records(predictions_path, claims_path, log_path)

    if df.empty:
        logger.error("No valid predictions found.")
        return

    y_true = df["gold_label"].tolist()
    y_pred = df["pred_label"].tolist()

    logger.info(f"Evaluating {len(df)} claims...")

    # Classification
    logger.info("Plotting classification metrics...")
    _plot_confusion_matrix(y_true, y_pred, out_cls)
    _plot_per_class_metrics(y_true, y_pred, out_cls)
    _plot_label_distribution(y_true, y_pred, out_cls)

    # Retrieval
    logger.info("Plotting retrieval metrics...")
    _plot_precision_recall_at_k(df, out_ret)
    _plot_score_distribution(df, out_ret)
    _plot_hit_rate_by_label(df, out_ret)

    # Error analysis
    logger.info("Plotting error analysis...")
    _plot_misclassification_heatmap(y_true, y_pred, out_err)
    _plot_score_vs_correctness(df, out_err)

    # Report
    logger.info("Writing text report...")
    _write_report(df, y_true, y_pred, out_base, log_path=log_path)

    logger.info(f"\nAll outputs saved to: {out_base}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SciFact pipeline results")
    parser.add_argument(
        "--model",
        choices=["default", "biomodel"],
        default="default",
        help="Embedding model configuration to evaluate (default: default)",
    )
    parser.add_argument(
        "--reranking",
        action="store_true",
        default=False,
        help="Evaluate the cross-encoder reranking configuration",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        default=False,
        dest="few_shot",
        help="Evaluate the few-shot prompt configuration",
    )
    args = parser.parse_args()
    evaluate(model=args.model, reranking=args.reranking, few_shot=args.few_shot)
