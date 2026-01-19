# Open RAG - Open-Source RAG System

A Retrieval-Augmented Generation (RAG) system supporting both open-source HuggingFace models and OpenAI's ChatGPT API.

## Features

- **Dual Generator Support**: Use HuggingFace open-source models OR OpenAI API
- **FAISS Retrieval**: Dense vector similarity search
- **BM25 Retrieval**: Sparse term-based retrieval
- **PDF Support**: Load and process PDF documents
- **Modular Architecture**: Mix and match retrievers with generators

## Structure

```
open_RAG/
├── main.py                     # CLI entry point
├── documents/                  # Place your documents here
└── src/
    ├── config.py               # Configuration settings
    ├── retrievers/             # Pure document retrieval
    │   ├── faiss_retriever.py  # FAISS dense retrieval
    │   └── bm25_retriever.py   # BM25 sparse retrieval
    └── generators/             # LLM generation
        ├── open_rag.py         # HuggingFace models
        └── gpt_rag.py          # OpenAI API
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # OR with uv
   uv sync
   ```

2. **Add your documents**:
   ```bash
   # Add .txt or .pdf files to the documents/ directory
   ```

3. **Set API keys** (in `.env` file):
   ```bash
   cp .env.example .env
   # Edit .env and add your tokens
   ```

## Usage

### Command Line Arguments

Run `python main.py --help` to see all available options.

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `query` | | The question or query to ask (Required) | |
| `--method` | `-m` | Retrieval method: `faiss` or `bm25` | `faiss` |
| `--generator` | `-g` | Generator model type: `huggingface` or `openai` | `huggingface` |
| `--model` | | Specific model name (e.g. `meta-llama/Llama-3.2-1B`, `gpt-4o`) | `gemma-2b-it` / `gpt-4o-mini` |
| `--documents` | `-d` | Path to documents directory | `./documents` |
| `--extension` | `-e` | File extension to load (`txt`, `pdf`, `md`) | `txt` |
| `--top-k` | `-k` | Number of document chunks to retrieve | `5` |
| `--max-tokens` | | Maximum new tokens to generate | `512` |
| `--temperature` | `-t` | Sampling temperature (0.0 to 1.0) | `0.6` |
| `--device` | | Device to use: `cpu`, `cuda`, `mps`, `auto` | `auto` |
| `--verbose` | `-v` | Enable verbose logging | `False` |

#### Examples

```bash
# 1. FAISS Retrieval + Open Source Model (HuggingFace)
python main.py "What is prompt injection?" --method faiss --generator huggingface --model meta-llama/Llama-3.2-1B

# 2. BM25 Retrieval + Open Source Model (HuggingFace)
python main.py "What is prompt injection?" --method bm25 --generator huggingface --model meta-llama/Llama-3.2-1B

# 3. FAISS Retrieval + OpenAI (GPT-4)
python main.py "How to defend?" --method faiss --generator openai --model gpt-4o-mini

# 4. BM25 Retrieval + OpenAI (GPT-4)
python main.py "How to defend?" --method bm25 --generator openai --model gpt-4o-mini
```

## Customizing the Prompt

Edit `PROMPT_TEMPLATE` in `src/retrievers/base.py`:

```python
PROMPT_TEMPLATE = """Your custom prompt here.

Context: {context}
Query: {query}
Response:"""
```

## Supported Models

### HuggingFace
- `google/gemma-2b-it` (default)
- `meta-llama/Llama-3.2-1B`
- `mistralai/Mistral-7B-Instruct-v0.1`
- Any HuggingFace causal LM

### OpenAI
- `gpt-4o-mini` (default)
- `gpt-4o`
- `gpt-4`
- `gpt-3.5-turbo`
