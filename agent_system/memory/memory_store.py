from __future__ import annotations

from typing import Iterable

import requests

from .milvus_store import MilvusMemoryStore
from .types import MemoryRecord, RetrievedMemory


class BaseMemoryStore:
    def initialize(self, mode: str, clean_before_init: bool = False) -> None:
        raise NotImplementedError

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        raise NotImplementedError

    def retrieve(self, query_text: str) -> list[RetrievedMemory]:
        raise NotImplementedError

    def export_jsonl(self, output_path: str, include_vectors: bool = True) -> int:
        raise NotImplementedError

    def rebuild_from_path(self, source_path: str, source_collection_name: str | None = None) -> int:
        raise NotImplementedError

    def update_records(self, updates: Iterable[dict]) -> int:
        raise NotImplementedError

    def prune_low_utility_memories(self, score_threshold: float, min_uses: int) -> int:
        return 0

    def clear_all_records(self) -> None:
        return

    def count_records(self) -> int:
        """Row count in the vector collection, or -1 if unsupported / unknown."""
        return -1

    def close(self) -> None:
        raise NotImplementedError


def _deserialize_retrieved_items(items: object) -> list[RetrievedMemory]:
    if not isinstance(items, list):
        return []

    retrieved: list[RetrievedMemory] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        retrieved.append(
            RetrievedMemory(
                memory_id=item.get("memory_id", ""),
                score=float(item.get("score", 0.0)),
                state_text=item.get("state_text", ""),
                action_text=item.get("action_text", ""),
                memory_text=item.get("memory_text", ""),
                reward=float(item.get("reward", 0.0)),
                metadata=metadata,
                value=item.get("value"),
                value_source=item.get("value_source"),
            )
        )
    return retrieved


class RemoteHTTPMemoryStore(BaseMemoryStore):
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        only_successful: bool = True,
        top_k: int = 3,
        min_score: float = 0.1,
    ) -> None:
        if not base_url:
            raise ValueError("`env.memory.vdb_base_url` must be set when using the HTTP memory backend.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.only_successful = only_successful
        self.top_k = top_k
        self.min_score = min_score

    def initialize(self, mode: str, clean_before_init: bool = False) -> None:
        del mode, clean_before_init

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        buffered = list(records)
        if not buffered:
            return

        response = requests.post(
            f"{self.base_url}/memories/batch",
            json={"items": [record.to_dict() for record in buffered]},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def retrieve(self, query_text: str) -> list[RetrievedMemory]:
        if not query_text:
            return []

        response = requests.post(
            f"{self.base_url}/memories/search",
            json={
                "query": query_text,
                "top_k": self.top_k,
                "min_score": self.min_score,
                "only_successful": self.only_successful,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _deserialize_retrieved_items(response.json())

    def export_jsonl(self, output_path: str, include_vectors: bool = True) -> int:
        del output_path, include_vectors
        raise NotImplementedError("HTTP memory backend does not support local export.")

    def rebuild_from_path(self, source_path: str, source_collection_name: str | None = None) -> int:
        del source_path, source_collection_name
        raise NotImplementedError("HTTP memory backend does not support local rebuild.")

    def update_records(self, updates: Iterable[dict]) -> int:
        buffered_updates = [dict(update) for update in updates if isinstance(update, dict)]
        if not buffered_updates:
            return 0

        response = requests.post(
            f"{self.base_url}/memories/update",
            json={"updates": buffered_updates},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return int(payload.get("updated", 0)) if isinstance(payload, dict) else 0

    def prune_low_utility_memories(self, score_threshold: float, min_uses: int) -> int:
        response = requests.post(
            f"{self.base_url}/memories/prune",
            json={"score_threshold": float(score_threshold), "min_uses": int(min_uses)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return int(payload.get("deleted", 0)) if isinstance(payload, dict) else 0

    def clear_all_records(self) -> None:
        response = requests.post(f"{self.base_url}/memories/clear", json={}, timeout=self.timeout)
        response.raise_for_status()

    def count_records(self) -> int:
        try:
            response = requests.get(f"{self.base_url}/memories/count", timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "count" in payload:
                return int(payload["count"])
        except Exception:
            pass
        return -1

    def close(self) -> None:
        return


class MemoryStoreDispatcher(BaseMemoryStore):
    def __init__(
        self,
        backend: str = "milvus",
        base_url: str | None = None,
        timeout: int = 30,
        only_successful: bool = True,
        top_k: int = 3,
        min_score: float = 0.1,
        task_name: str = "memory",
        store_dir: str | None = None,
        collection_name: str | None = None,
        embedding_api_url: str | None = None,
        embedding_api_key: str = "empty",
        embedding_model: str = "bge_m3",
        embedding_dim: int = 1024,
        retrieve_key: str = "memory_text",
        rebuild_source_path: str | None = None,
        rebuild_source_collection_name: str | None = None,
        rebuild_insert_batch_size: int = 1000,
        rebuild_embedding_batch_size: int = 256,
        memory_text_retrieval_dedupe_similarity_threshold: float | None = None,
        memory_text_retrieval_dedupe_prefetch_limit: int = 24,
        memory_text_insert_dedupe_similarity_threshold: float | None = None,
        memory_text_insert_dedupe_probe_limit: int = 40,
    ) -> None:
        normalized_backend = (backend or "").strip().lower()
        if not normalized_backend:
            normalized_backend = "http" if base_url else "milvus"

        if normalized_backend == "http":
            self.backend = RemoteHTTPMemoryStore(
                base_url=base_url or "",
                timeout=timeout,
                only_successful=only_successful,
                top_k=top_k,
                min_score=min_score,
            )
            return

        if normalized_backend != "milvus":
            raise ValueError(f"Unsupported memory backend: {backend}")

        self.backend = MilvusMemoryStore(
            task_name=task_name,
            store_dir=store_dir or "memory_store",
            collection_name=collection_name,
            embedding_api_url=embedding_api_url,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            timeout=timeout,
            only_successful=only_successful,
            top_k=top_k,
            min_score=min_score,
            retrieve_key=retrieve_key,
            rebuild_source_path=rebuild_source_path,
            rebuild_source_collection_name=rebuild_source_collection_name,
            rebuild_insert_batch_size=rebuild_insert_batch_size,
            rebuild_embedding_batch_size=rebuild_embedding_batch_size,
            memory_text_retrieval_dedupe_similarity_threshold=memory_text_retrieval_dedupe_similarity_threshold,
            memory_text_retrieval_dedupe_prefetch_limit=memory_text_retrieval_dedupe_prefetch_limit,
            memory_text_insert_dedupe_similarity_threshold=memory_text_insert_dedupe_similarity_threshold,
            memory_text_insert_dedupe_probe_limit=memory_text_insert_dedupe_probe_limit,
        )

    def initialize(self, mode: str, clean_before_init: bool = False) -> None:
        self.backend.initialize(mode=mode, clean_before_init=clean_before_init)

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        self.backend.add_records(records)

    def retrieve(self, query_text: str) -> list[RetrievedMemory]:
        return self.backend.retrieve(query_text=query_text)

    def export_jsonl(self, output_path: str, include_vectors: bool = True) -> int:
        return self.backend.export_jsonl(output_path=output_path, include_vectors=include_vectors)

    def rebuild_from_path(self, source_path: str, source_collection_name: str | None = None) -> int:
        return self.backend.rebuild_from_path(source_path=source_path, source_collection_name=source_collection_name)

    def update_records(self, updates: Iterable[dict]) -> int:
        return self.backend.update_records(updates)

    def prune_low_utility_memories(self, score_threshold: float, min_uses: int) -> int:
        return self.backend.prune_low_utility_memories(score_threshold, min_uses)

    def clear_all_records(self) -> None:
        return self.backend.clear_all_records()

    def count_records(self) -> int:
        return self.backend.count_records()

    def close(self) -> None:
        self.backend.close()
