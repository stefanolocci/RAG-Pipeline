"""
Gemini client for SciFact verification pipeline.

Exposes two public functions used by the rest of the pipeline:

    embed_texts(texts, task_type) -> list[list[float]]   (local SentenceTransformer)
    generate(prompt)              -> str                  (Gemini web automation)
"""

import logging

from sentence_transformers import SentenceTransformer, models as st_models

from src.config import GENERATION_MODEL, SENTENCE_MODEL_NAME, BIO_EMBEDDING_MODEL
from src.generators.gemini_api import GeminiAutomator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised)
# ---------------------------------------------------------------------------

# Cache of loaded ST models keyed by HuggingFace model name
_st_models: dict[str, SentenceTransformer] = {}
_automator: GeminiAutomator | None = None


def _get_st_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer for the given model name.

    For non-native ST models (e.g. plain BERT checkpoints), the model is
    wrapped with a mean-pooling layer automatically.
    """
    if model_name not in _st_models:
        logger.info(f"Loading embedding model: {model_name}")
        try:
            # Try loading as a native sentence-transformers model first
            _st_models[model_name] = SentenceTransformer(model_name)
        except Exception:
            # Fall back: wrap a plain HF Transformer with mean pooling
            transformer = st_models.Transformer(model_name)
            pooling = st_models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
            )
            _st_models[model_name] = SentenceTransformer(modules=[transformer, pooling])
        logger.info(f"Model ready: {model_name}")
    return _st_models[model_name]


def _get_automator() -> GeminiAutomator:
    global _automator
    if _automator is None:
        logger.info("Initialising Gemini web automator...")
        _automator = GeminiAutomator()
        if not _automator.initialize():
            raise RuntimeError(
                "GeminiAutomator failed to initialise. "
                "Check that GEMINI_SECURE_1PSID and GEMINI_SECURE_1PSIDTS are set in .env"
            )
    return _automator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    model_name: str | None = None,
) -> list[list[float]]:
    """
    Embed texts locally with SentenceTransformer.

    Args:
        texts:      Strings to embed.
        task_type:  Accepted for interface compatibility, ignored locally.
        model_name: HuggingFace model name to use. Defaults to SENTENCE_MODEL_NAME.

    Returns:
        List of embedding vectors.
    """
    name = model_name or SENTENCE_MODEL_NAME
    st_model = _get_st_model(name)
    logger.info(f"Embedding {len(texts)} texts with {name}...")
    embeddings = st_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def generate(prompt: str) -> str:
    """
    Generate text via the Gemini web automator (no API key required).

    Args:
        prompt: Full prompt string.

    Returns:
        Model response as a string.

    Raises:
        RuntimeError: If the automator fails after retries.
    """
    automator = _get_automator()
    for attempt in range(3):
        result = automator.generate_text(prompt, model=GENERATION_MODEL)
        if result is not None:
            return result
        logger.warning(f"generate_text returned None (attempt {attempt + 1}/3), retrying...")
    raise RuntimeError("Gemini web automator failed to return a response after 3 attempts")
