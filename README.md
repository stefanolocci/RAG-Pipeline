# RAG Pipeline

A modular **Retrieval-Augmented Generation (RAG)** system with two independent pipelines:

- **Generic RAG** — bring your own documents, choose your retriever and LLM
- **SciFact Verification** — biomedical claim verification pipeline evaluated on the [SciFact](https://github.com/allenai/scifact) dataset

---

## Table of Contents

1. [Installation](#installation)
3. [Generic RAG Pipeline](#generic-rag-pipeline)
4. [SciFact Verification Pipeline](#scifact-verification-pipeline)
   - [Architecture](#architecture)
   - [Data](#data)
   - [Step 1 — Build the Index](#step-1--build-the-index)
   - [Step 2 — Run the Pipeline](#step-2--run-the-pipeline)
   - [Step 3 — Evaluate](#step-3--evaluate)
   - [Embedding Models](#embedding-models)
   - [Reranking](#reranking)
   - [Results](#results)
5. [Configuration](#configuration)
6. [Environment Variables](#environment-variables)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/RAG-Pipeline.git
cd RAG-Pipeline

# Install with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

**Required Python:** 3.12+

---

## Generic RAG Pipeline

Query your own documents using FAISS or BM25 retrieval combined with a local HuggingFace model or OpenAI API.

### Setup

1. Place your `.txt` or `.pdf` files in `documents/`
2. Configure API keys in `.env` (see [Environment Variables](#environment-variables))

### Usage

```bash
python main.py "Your question here" [options]
```

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `query` | | Question to ask (required) | |
| `--method` | `-m` | `faiss` or `bm25` | `faiss` |
| `--generator` | `-g` | `huggingface` or `openai` | `huggingface` |
| `--model` | | HuggingFace or OpenAI model name | `gemma-2b-it` |
| `--documents` | `-d` | Path to documents directory | `./documents` |
| `--extension` | `-e` | File type to load: `txt`, `pdf`, `md` | `txt` |
| `--top-k` | `-k` | Number of chunks to retrieve | `5` |
| `--max-tokens` | | Max tokens to generate | `512` |
| `--temperature` | `-t` | Sampling temperature | `0.6` |
| `--device` | | `cpu`, `cuda`, `mps`, `auto` | `auto` |
| `--verbose` | `-v` | Verbose logging | `False` |

## SciFact Verification Pipeline

End-to-end system for **biomedical claim verification** using the SciFact corpus (~5,000 abstracts). Given a claim, the system retrieves relevant abstracts and classifies it as `SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFO`.

### Architecture

```
┌─────────────┐    ┌───────────────────────┐    ┌──────────────────────┐
│    CLAIM    │───▶│      RETRIEVER        │───▶│     GENERATOR        │
│   (query)   │    │  FAISS + ST embedding │    │  Gemini-2.5-flash    │
└─────────────┘    │  (optional reranking) │    │  classify + explain  │
                   └───────────────────────┘    └──────────────────────┘
                            │                             │
                   ┌────────┴────────┐           ┌───────┴────────┐
                   │  SciFact corpus │           │  label +       │
                   │  (~5k abstracts)│           │  explanation   │
                   └─────────────────┘           └────────────────┘
```

### Data

Download the SciFact dataset and place it in `data/`:

```bash
wget https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
tar -xvzf data.tar.gz -C data/ --strip-components=1
```

Expected files:
```
data/
├── corpus.jsonl        # ~5,183 abstracts
├── claims_dev.jsonl    # 300 annotated claims
└── claims_train.jsonl  # ~809 claims
```

### Step 1 — Build the Index

Build the FAISS index once per embedding model. Embeddings are computed **locally** (no API calls).

```bash
# Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
uv run src/indexing.py

```

Index files are saved to `indices/` and reused on subsequent runs.

### Step 2 — Run the Pipeline

```bash
# Default model, no reranking (~75 min for 300 claims at 4 RPM)
uv run src/run_pipeline.py

# Default model + cross-encoder reranking
uv run src/run_pipeline.py --reranking
```

Each configuration writes to its own output files so runs never overwrite each other:

| Flags | Predictions | Log |
|-------|-------------|-----|
| *(none)* | `outputs/predictions.jsonl` | `outputs/detailed_log.jsonl` |
| `--reranking` | `outputs/predictions_reranking.jsonl` | `outputs/detailed_log_reranking.jsonl` |

**Resume support:** interrupted runs resume automatically, retrying only failed claims.

### Step 3 — Evaluate

```bash
uv run src/evaluate.py
```
> See the full analysis report: [evaluation/standard_model/report_italiano.md](evaluation/standard_model/report_italiano.md)

### Embedding Models

| Key | Model | Dim | Notes |
|-----|-------|----:|-------|
| `default` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, general-purpose |

The model runs **entirely locally** — no API calls for embeddings.

### Reranking

When `--reranking` is passed, after FAISS retrieves 20 candidates, a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each `(claim, abstract)` pair and re-orders them by relevance before passing the top-5 to the LLM. This improves precision at the cost of ~2s extra per claim.

### Results

Results from the baseline configuration (default model, no reranking) on 300 dev claims:

| Metric | Value |
|--------|------:|
| **Accuracy** | **0.780** |
| **Macro F1** | **0.777** |
| Precision@5 | 0.187 |
| Recall@5 | 0.869 |
| Hit Rate | 0.878 |

| Class | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| `SUPPORTS` | 0.78 | 0.87 | 0.82 |
| `REFUTES` | 0.72 | 0.91 | 0.81 |
| `NOT_ENOUGH_INFO` | 0.84 | 0.61 | 0.70 |

> Full analysis with charts and commentary: [evaluation/standard_model/report_italiano.md](evaluation/standard_model/report_italiano.md)

---

## Configuration

All settings live in `src/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `SENTENCE_MODEL_NAME` | `all-MiniLM-L6-v2` | Default embedding model |
| `GENERATION_MODEL` | `gemini-2.5-flash` | LLM for classification |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `TOP_K` | `5` | Documents returned to LLM |
| `RERANKING_CANDIDATES_K` | `20` | FAISS candidates before reranking |
| `CORPUS_PATH` | `data/corpus.jsonl` | SciFact corpus |
| `CLAIMS_DEV_PATH` | `data/claims_dev.jsonl` | Dev set |

---

## Environment Variables

Create a `.env` file in the project root:

```env
# HuggingFace (required for gated models like Llama)
HF_TOKEN=hf_...

# OpenAI (required for --generator openai)
OPENAI_API_KEY=sk-...

# Google Gemini API (optional — used if not using web automation)
GOOGLE_API_KEY=...

# Gemini web automation cookies (alternative to API key)
# Get from browser DevTools → gemini.google.com → Cookies
GEMINI_SECURE_1PSID=...
GEMINI_SECURE_1PSIDTS=...
```
