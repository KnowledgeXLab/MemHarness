# Copyright 2025 MemAdaptor
"""Embedding-based cosine similarity between memory historical state and current state."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from agent_system.memory.memory_text_dedupe import cosine_similarity_float_vectors
from agent_system.memory.memory_manager import normalize_text
from agent_system.memory.milvus_store import RemoteEmbeddingProvider

_CLIENT_CACHE: dict[tuple, RemoteEmbeddingProvider] = {}


def _memory_embedding_provider(config: DictConfig) -> Optional[RemoteEmbeddingProvider]:
    mem = OmegaConf.select(config, "env.memory")
    if mem is None:
        return None
    api_url = str(mem.get("embedding_api_url") or "").strip()
    if not api_url:
        return None
    key = (
        api_url,
        str(mem.get("embedding_api_key") or "empty"),
        str(mem.get("embedding_model") or "bge_m3"),
        int(mem.get("embedding_dim") or 1024),
        int(mem.get("vdb_timeout") or 30),
    )
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = RemoteEmbeddingProvider(
            api_url=api_url,
            api_key=str(mem.get("embedding_api_key") or "empty"),
            model_name=str(mem.get("embedding_model") or "bge_m3"),
            embedding_dim=int(mem.get("embedding_dim") or 1024),
            timeout=int(mem.get("vdb_timeout") or 30),
        )
    return _CLIENT_CACHE[key]


def _clip_state_text(config: DictConfig, text: str) -> str:
    mem = OmegaConf.select(config, "env.memory") or {}
    max_chars = int(mem.get("max_state_chars") or 1200)
    return normalize_text(text or "", max_chars=max_chars)


def batch_state_cosine_similarities(
    config: DictConfig,
    states_old: Sequence[str],
    states_curr: Sequence[str],
) -> list[float]:
    """Cosine similarity between paired historical and current states (same embedding API as VDB)."""
    if len(states_old) != len(states_curr):
        raise ValueError("states_old and states_curr must have the same length")
    n = len(states_old)
    if n == 0:
        return []

    provider = _memory_embedding_provider(config)
    if provider is None:
        return [float("nan")] * n

    out: list[float] = [float("nan")] * n
    embed_indices: list[int] = []
    embed_texts: list[str] = []
    for i, (s_old, s_curr) in enumerate(zip(states_old, states_curr)):
        old_t = _clip_state_text(config, s_old)
        cur_t = _clip_state_text(config, s_curr)
        if not old_t.strip() or not cur_t.strip():
            continue
        embed_indices.extend([i, i])
        embed_texts.extend([old_t, cur_t])

    if not embed_texts:
        return out

    try:
        vectors = provider.get_embeddings(embed_texts)
    except Exception:
        return out

    for j, idx in enumerate(range(0, len(embed_texts), 2)):
        pair_i = embed_indices[j * 2]
        va = vectors[j * 2]
        vb = vectors[j * 2 + 1]
        if not va or not vb:
            continue
        sim = cosine_similarity_float_vectors(va, vb)
        if math.isfinite(sim):
            out[pair_i] = float(sim)
    return out
