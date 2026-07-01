"""Local embeddings via fastembed.

Using a local model (BAAI/bge-small-en-v1.5) means embeddings need no API key
and stay reproducible across runs — see README "Stack rationale". The model is
loaded lazily and cached process-wide.
"""
import os
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import settings


@lru_cache
def _model() -> TextEmbedding:
    # Cache the model on a persistent volume so it downloads only once.
    cache_dir = os.environ.get("FASTEMBED_CACHE_PATH") or None
    return TextEmbedding(model_name=settings.embedding_model, cache_dir=cache_dir)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    # batch_size keeps memory bounded on large reports.
    return [vec.tolist() for vec in _model().embed(texts, batch_size=32)]


def embed_query(text: str) -> list[float]:
    return next(_model().query_embed(text)).tolist()
