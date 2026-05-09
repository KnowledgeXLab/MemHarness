# Copyright 2025 MemAdaptor
"""Embedding-space deduplication helpers for ``memory_text`` (or any aligned text vectors)."""

from __future__ import annotations

import math
from typing import Sequence


def cosine_similarity_float_vectors(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for dense float vectors (same length)."""
    if not a or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def dedupe_indices_by_embedding_similarity(embeddings: list[list[float]], threshold: float) -> list[int]:
    """Greedy keep-first ordering: drop items whose embedding is >= threshold similar to any kept one."""
    if threshold <= 0.0:
        return list(range(len(embeddings)))
    kept_idx: list[int] = []
    kept_vecs: list[list[float]] = []
    for i, emb in enumerate(embeddings):
        if any(cosine_similarity_float_vectors(emb, kv) >= threshold for kv in kept_vecs):
            continue
        kept_idx.append(i)
        kept_vecs.append(emb)
    return kept_idx
