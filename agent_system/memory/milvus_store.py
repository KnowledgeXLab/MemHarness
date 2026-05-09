from __future__ import annotations

import atexit
import json
import logging
import os
import re
import time
from typing import Iterable

from .experience_utility import (
    UTILITY_SCORE_KEY,
    UTILITY_SUCC_KEY,
    UTILITY_USE_KEY,
    compute_utility_score,
    read_utility_score_from_metadata,
)
from .memory_text_dedupe import cosine_similarity_float_vectors, dedupe_indices_by_embedding_similarity
from .types import MemoryRecord, RetrievedMemory

SUPPORTED_MEMORY_INIT_MODES = {
    "init_if_missing",
    "recreate",
}


def _safe_str(value: object, max_length: int) -> str:
    text = "" if value is None else str(value)
    if max_length > 0 and len(text) > max_length:
        return text[:max_length]
    return text


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _normalize_collection_name(task_name: str, collection_name: str | None) -> str:
    if collection_name:
        return collection_name
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", task_name or "memory").strip("_")
    return f"agent_memories_{normalized or 'memory'}"


class RemoteEmbeddingProvider:
    """OpenAI-compatible embedding client used by the local vector store."""

    def __init__(
        self,
        api_url: str,
        api_key: str = "empty",
        model_name: str = "bge_m3",
        embedding_dim: int = 1024,
        timeout: int = 30,
    ) -> None:
        if not api_url:
            raise ValueError("`env.memory.embedding_api_url` must be set when memory is enabled.")
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.timeout = timeout
        self.client = None

    def _ensure_client(self):
        if self.client is not None:
            return self.client

        import httpx
        from openai import OpenAI

        http_client = httpx.Client(verify=False, trust_env=False, timeout=float(self.timeout))
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url, http_client=http_client)
        return self.client

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._ensure_client()
        embedding_obj = client.embeddings.create(input=texts, model=self.model_name)
        embeddings = [list(item.embedding) for item in embedding_obj.data]
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding service returned {len(embeddings)} vectors for {len(texts)} inputs."
            )
        return embeddings


class MilvusMemoryStore:
    """Local Milvus-Lite backed vector memory store

    Initialization modes (``env.memory.mode``):
    - ``init_if_missing``: Open or create the local ``milvus_memory.db``. If the target
      collection already exists, **do not** reload from ``rebuild_source_path`` even if it
      points to a new ``.jsonl``; use ``recreate`` or ``clean_before_init`` to force rebuild.
    - ``recreate``: Drop the target collection if present, then rebuild from
      ``rebuild_source_path`` when that path is set (``.jsonl`` or ``.db``).
    """

    def __init__(
        self,
        task_name: str,
        store_dir: str,
        collection_name: str | None = None,
        embedding_api_url: str | None = None,
        embedding_api_key: str = "empty",
        embedding_model: str = "bge_m3",
        embedding_dim: int = 1024,
        timeout: int = 30,
        only_successful: bool = True,
        top_k: int = 3,
        min_score: float = 0.1,
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
        self.task_name = task_name
        self.store_dir = os.path.abspath(store_dir)
        self.db_path = os.path.join(self.store_dir, "milvus_memory.db")
        self.log_path = os.path.join(self.store_dir, "milvus_memory.log")
        self.collection_name = _normalize_collection_name(task_name=task_name, collection_name=collection_name)
        self.only_successful = only_successful
        self.top_k = top_k
        self.min_score = min_score
        self.timeout = timeout
        self.embedding_dim = embedding_dim
        self.retrieve_key = retrieve_key
        # Optional path to seed the DB from a .jsonl export or another .db (see initialize()).
        self.rebuild_source_path = rebuild_source_path
        self.rebuild_source_collection_name = rebuild_source_collection_name
        self.rebuild_insert_batch_size = max(1, int(rebuild_insert_batch_size))
        self.rebuild_embedding_batch_size = max(1, int(rebuild_embedding_batch_size))
        self.memory_text_retrieval_dedupe_similarity_threshold = self._normalize_similarity_threshold(
            memory_text_retrieval_dedupe_similarity_threshold
        )
        self.memory_text_retrieval_dedupe_prefetch_limit = max(1, int(memory_text_retrieval_dedupe_prefetch_limit or 24))
        self.memory_text_insert_dedupe_similarity_threshold = self._normalize_similarity_threshold(
            memory_text_insert_dedupe_similarity_threshold
        )
        self.memory_text_insert_dedupe_probe_limit = max(1, int(memory_text_insert_dedupe_probe_limit or 40))
        self.embedding_provider = RemoteEmbeddingProvider(
            api_url=embedding_api_url or "",
            api_key=embedding_api_key,
            model_name=embedding_model,
            embedding_dim=embedding_dim,
            timeout=timeout,
        )

        self._client = None
        self._milvus_client_cls = None
        self._collection_schema_cls = None
        self._field_schema_cls = None
        self._data_type = None

        os.makedirs(self.store_dir, exist_ok=True)
        self.logger = self._build_logger()
        atexit.register(self.close)

    @staticmethod
    def _normalize_similarity_threshold(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= threshold <= 1.0):
            logging.getLogger(__name__).warning(
                "memory_text dedupe similarity threshold must be in [0, 1]; ignoring %s",
                value,
            )
            return None
        return threshold

    def _task_name_filter_expr(self) -> str:
        escaped = self.task_name.replace("\\", "\\\\").replace('"', '\\"')
        return f'task_name == "{escaped}"'

    def _scoped_search_filter_expr(self) -> str:
        parts = [self._task_name_filter_expr()]
        if self.only_successful:
            parts.append("success == true")
        return " && ".join(parts)

    def _neighbor_has_similar_memory_text(
        self,
        client,
        retrieve_vector: list[float],
        memory_vec: list[float],
        threshold: float,
    ) -> bool:
        results = client.search(
            collection_name=self.collection_name,
            data=[retrieve_vector],
            limit=int(self.memory_text_insert_dedupe_probe_limit),
            output_fields=["memory_text"],
            filter=self._scoped_search_filter_expr(),
        )
        hits = results[0] if results else []
        texts: list[str] = []
        for hit in hits:
            entity = hit.get("entity", {})
            raw = entity.get("memory_text", "")
            if isinstance(raw, str) and raw.strip():
                texts.append(raw)
        if not texts:
            return False
        neighbor_embeddings = self._embed_texts(texts)
        return any(cosine_similarity_float_vectors(memory_vec, nb) >= threshold for nb in neighbor_embeddings)

    def initialize(self, mode: str, clean_before_init: bool = False) -> None:
        """Attach to or create the local DB and optionally load vectors from ``rebuild_source_path``.

        Rebuild from ``rebuild_source_path`` occurs only when that path is set **and**
        (``mode == recreate`` **or** the target collection is missing). With
        ``init_if_missing`` and an existing collection, a new ``.jsonl`` path alone does
        **not** trigger reload; set ``mode=recreate`` or ``clean_before_init=True``.
        """
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in SUPPORTED_MEMORY_INIT_MODES:
            raise ValueError(
                f"Unsupported memory mode: {mode}. Expected one of {sorted(SUPPORTED_MEMORY_INIT_MODES)}."
            )

        client = self._ensure_client()
        self.logger.info(
            "initialize store_dir=%s db_path=%s collection=%s mode=%s clean_before_init=%s rebuild_source_path=%s",
            self.store_dir,
            self.db_path,
            self.collection_name,
            normalized_mode,
            clean_before_init,
            self.rebuild_source_path,
        )

        # Delete local Milvus-Lite files sharing the same stem (e.g. milvus_memory.db*) so the next
        # client opens a fresh store; use when you want to force a full re-import from rebuild_source_path.
        if clean_before_init and os.path.isdir(self.store_dir):
            self.close()
            db_stem = os.path.splitext(os.path.basename(self.db_path))[0]
            removed: list[str] = []
            for entry in os.listdir(self.store_dir):
                if entry.startswith(db_stem):
                    path = os.path.join(self.store_dir, entry)
                    if os.path.isfile(path):
                        os.remove(path)
                        removed.append(path)
            if removed:
                self.logger.info(
                    "clean_before_init removed %s file(s): %s",
                    len(removed),
                    removed,
                )
            else:
                self.logger.info("clean_before_init: no files matched prefix '%s' under %s", db_stem, self.store_dir)
            client = self._ensure_client(force_new=True)

        collections = set(client.list_collections())
        if normalized_mode == "recreate" and self.collection_name in collections:
            client.drop_collection(self.collection_name)
            collections.remove(self.collection_name)
            self.logger.info(
                "initialize mode=recreate dropped existing collection=%s before optional rebuild",
                self.collection_name,
            )

        # Rebuild when a source path is configured and either:
        # - recreate: always reload from source after the drop above; or
        # - init_if_missing: only when the collection is still absent (new DB or after clean_before_init).
        # Otherwise init_if_missing keeps the existing collection and ignores a new .jsonl path.
        should_rebuild = bool(self.rebuild_source_path) and (
            normalized_mode == "recreate" or self.collection_name not in collections
        )
        self.logger.info(
            "rebuild decision: should_rebuild=%s mode=%s collection_in_store=%s collection=%s",
            should_rebuild,
            normalized_mode,
            self.collection_name in collections,
            self.collection_name,
        )
        if should_rebuild:
            self.logger.info(
                "rebuilding vector store from source path=%s (mode=%s)",
                self.rebuild_source_path,
                normalized_mode,
            )
            self.rebuild_from_path(
                source_path=self.rebuild_source_path,
                source_collection_name=self.rebuild_source_collection_name,
            )
            return

        if self.rebuild_source_path and not should_rebuild:
            self.logger.info(
                "skipping rebuild from rebuild_source_path=%s: collection %s already exists and "
                "mode=%s (use mode=recreate or clean_before_init=True to reload from JSONL/DB)",
                self.rebuild_source_path,
                self.collection_name,
                normalized_mode,
            )

        if self.collection_name not in collections:
            self._create_collection()
            self.logger.info("created empty collection=%s (no rebuild from source)", self.collection_name)

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        buffered = list(records)
        if not buffered:
            return

        retrieve_texts = [_safe_str(getattr(record, self.retrieve_key, ""), 8192) for record in buffered]
        embeddings = self._embed_texts(retrieve_texts)
        insert_thr = self.memory_text_insert_dedupe_similarity_threshold
        memory_embeddings: list[list[float]] | None = None
        if insert_thr is not None:
            memory_embeddings = self._embed_texts([_safe_str(record.memory_text, 8192) for record in buffered])

        entities: list[dict] = []
        batch_kept_memory_vecs: list[list[float]] = []
        skipped_similarity = 0
        client = self._ensure_client() if insert_thr is not None else None
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        vector_field_name = self._vector_field_name

        for idx, record in enumerate(buffered):
            mem_plain = _safe_str(record.memory_text, 8192)
            if insert_thr is not None and memory_embeddings is not None and client is not None and mem_plain.strip():
                vec_m = memory_embeddings[idx]
                if any(cosine_similarity_float_vectors(vec_m, kv) >= insert_thr for kv in batch_kept_memory_vecs):
                    skipped_similarity += 1
                    continue
                if self._neighbor_has_similar_memory_text(client, embeddings[idx], vec_m, insert_thr):
                    skipped_similarity += 1
                    continue
                batch_kept_memory_vecs.append(vec_m)

            entities.append(
                {
                    "memory_id": _safe_str(record.memory_id, 128),
                    "task_name": _safe_str(record.task_name, 128),
                    "item_id": int(record.item_id),
                    "source_episode_id": _safe_str(record.source_episode_id, 128),
                    "source_step": int(record.source_step),
                    vector_field_name: embeddings[idx],
                    "state_text": _safe_str(record.state_text, 4096),
                    "action_text": _safe_str(record.action_text, 4096),
                    "memory_text": mem_plain,
                    "reward": _safe_float(record.reward),
                    "success": bool(record.success),
                    "created_step": _safe_str(record.created_step, 64),
                    "retrieval_count": int(record.retrieval_count),
                    "last_used_step": _safe_str(record.last_used_step, 64),
                    "metadata": _safe_str(json.dumps(record.metadata, ensure_ascii=False), 8192),
                    "value": _safe_float(record.value),
                    "value_source": _safe_str(record.value_source, 64),
                    "value_update_step": _safe_str(record.value_update_step, 64),
                    "created_at": _safe_str(record.created_at or now_ts, 64),
                }
            )

        inserted = self._insert_entities_in_batches(entities)
        self.logger.info(
            "add_records inserted=%s skipped_similarity=%s collection=%s",
            inserted,
            skipped_similarity,
            self.collection_name,
        )

    def retrieve(self, query_text: str) -> list[RetrievedMemory]:
        if not query_text:
            return []

        client = self._ensure_client()
        query_vector = self._embed_texts([query_text])[0]
        recall_thr = self.memory_text_retrieval_dedupe_similarity_threshold
        search_limit = max(1, self.top_k)
        if recall_thr is not None:
            search_limit = max(search_limit, int(self.memory_text_retrieval_dedupe_prefetch_limit))

        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "limit": search_limit,
            "output_fields": [
                "memory_id",
                "state_text",
                "action_text",
                "memory_text",
                "reward",
                "metadata",
                "value",
                "value_source",
            ],
        }
        if self.only_successful:
            search_kwargs["filter"] = "success == true"

        results = client.search(**search_kwargs)
        hits = results[0] if results else []
        retrieved: list[RetrievedMemory] = []
        for hit in hits:
            entity = hit.get("entity", {})
            score = _safe_float(1 - hit["distance"])
            if score < self.min_score:
                continue
            metadata = entity.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {"raw_metadata": metadata}
            retrieved.append(
                RetrievedMemory(
                    memory_id=entity.get("memory_id", ""),
                    score=score,
                    state_text=entity.get("state_text", ""),
                    action_text=entity.get("action_text", ""),
                    memory_text=entity.get("memory_text", ""),
                    reward=_safe_float(entity.get("reward", 0.0)),
                    metadata=metadata if isinstance(metadata, dict) else {},
                    value=entity.get("value"),
                    value_source=entity.get("value_source"),
                )
            )
        before_dedupe = len(retrieved)
        if recall_thr is not None and len(retrieved) > 1:
            texts_for_emb = [r.memory_text if r.memory_text.strip() else " " for r in retrieved]
            mem_vecs = self._embed_texts(texts_for_emb)
            keep_ix = dedupe_indices_by_embedding_similarity(mem_vecs, recall_thr)
            retrieved = [retrieved[i] for i in keep_ix]
        trimmed = retrieved[: max(1, self.top_k)]
        self.logger.info(
            "retrieve collection=%s top_k=%s returned=%s after_dedupe=%s final=%s prefetch=%s min_score=%.4f query=%s",
            self.collection_name,
            self.top_k,
            before_dedupe,
            len(retrieved),
            len(trimmed),
            search_limit,
            self.min_score,
            _safe_str(query_text, 50) + "..." if len(query_text) > 50 else query_text,
        )
        return trimmed

    def rebuild_from_path(self, source_path: str, source_collection_name: str | None = None) -> int:
        """Load vectors into this store from a ``.jsonl`` export or another Milvus-Lite ``.db``.

        Returns the number of rows inserted (see ``_rebuild_from_jsonl`` / ``_rebuild_from_db``).
        """
        source_path = os.path.abspath(source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Memory rebuild source does not exist: {source_path}")
        self.logger.info(
            "rebuild_from_path start source=%s source_collection=%s target_collection=%s target_db=%s",
            source_path,
            source_collection_name,
            self.collection_name,
            self.db_path,
        )

        if source_path.endswith(".jsonl"):
            return self._rebuild_from_jsonl(source_path)
        if source_path.endswith(".db"):
            return self._rebuild_from_db(source_path, source_collection_name=source_collection_name)
        raise ValueError(f"Unsupported rebuild source: {source_path}")

    def export_jsonl(self, output_path: str, include_vectors: bool = True) -> int:
        client = self._ensure_client()
        records = self._iter_collection_records_from_client(
            client=client,
            collection_name=self.collection_name,
            include_vectors=include_vectors,
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        exported = 0
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1
        self.logger.info("export_jsonl exported=%s output=%s", exported, output_path)
        return exported

    def update_records(self, updates: Iterable[dict]) -> int:
        buffered_updates = [dict(update) for update in updates if update and update.get("memory_id")]
        if not buffered_updates:
            return 0

        client = self._ensure_client()
        vector_field_name = self._vector_field_name
        updated_entities: list[dict] = []
        updated_ids: list[str] = []
        for update in buffered_updates:
            memory_id = str(update["memory_id"])
            existing_rows = client.query(
                collection_name=self.collection_name,
                filter=f'memory_id == "{memory_id}"',
                output_fields=[
                    "memory_id",
                    "task_name",
                    "item_id",
                    "source_episode_id",
                    "source_step",
                    vector_field_name,
                    "state_text",
                    "action_text",
                    "memory_text",
                    "reward",
                    "success",
                    "created_step",
                    "retrieval_count",
                    "last_used_step",
                    "metadata",
                    "value",
                    "value_source",
                    "value_update_step",
                    "created_at",
                ],
                limit=1,
            )
            if not existing_rows:
                continue

            row = dict(existing_rows[0])
            retrieval_increment = int(update.get("_increment_retrieval_count", 0))

            current_metadata = row.get("metadata", {})
            if isinstance(current_metadata, str):
                try:
                    current_metadata = json.loads(current_metadata)
                except json.JSONDecodeError:
                    current_metadata = {}
            if not isinstance(current_metadata, dict):
                current_metadata = {}
            merged_metadata = dict(current_metadata)

            meta_inc = update.get("_metadata_int_increment")
            if meta_inc and isinstance(meta_inc, dict):
                for key, delta in meta_inc.items():
                    merged_metadata[key] = int(merged_metadata.get(key, 0)) + int(delta)

            if "metadata" in update and isinstance(update["metadata"], dict):
                merged_metadata.update(update["metadata"])

            if UTILITY_USE_KEY in merged_metadata or UTILITY_SUCC_KEY in merged_metadata:
                c_use = int(merged_metadata.get(UTILITY_USE_KEY, 0))
                c_succ = int(merged_metadata.get(UTILITY_SUCC_KEY, 0))
                u_score = compute_utility_score(c_use, c_succ)
                merged_metadata[UTILITY_SCORE_KEY] = u_score
                if bool(update.get("_sync_value_from_utility", False)):
                    row["value"] = float(u_score)
                    row["value_source"] = "utility_score"
                    vu = update.get("value_update_step")
                    if vu is not None:
                        row["value_update_step"] = _safe_str(vu, 64)

            row["metadata"] = merged_metadata

            for field in [
                "task_name",
                "item_id",
                "source_episode_id",
                "source_step",
                "state_text",
                "action_text",
                "memory_text",
                "reward",
                "success",
                "created_step",
                "retrieval_count",
                "last_used_step",
                "value",
                "value_source",
                "value_update_step",
                "created_at",
            ]:
                if field in update:
                    row[field] = update[field]

            if retrieval_increment:
                row["retrieval_count"] = int(row["retrieval_count"]) + retrieval_increment

            if self.retrieve_key in update and vector_field_name not in update:
                row[vector_field_name] = self._embed_texts([str(row[self.retrieve_key])])[0]
            elif vector_field_name in update:
                row[vector_field_name] = update[vector_field_name]

            updated_entities.append(self._entity_from_row(row))
            updated_ids.append(memory_id)

        if not updated_entities:
            return 0

        expr = ", ".join(f'"{memory_id}"' for memory_id in updated_ids)
        client.delete(collection_name=self.collection_name, filter=f"memory_id in [{expr}]")
        self._insert_entities_in_batches(updated_entities)
        self.logger.info("update_records updated=%s collection=%s", len(updated_entities), self.collection_name)
        return len(updated_entities)

    def prune_low_utility_memories(self, score_threshold: float, min_uses: int) -> int:
        """Delete memories whose utility score is below ``score_threshold`` after ``min_uses`` retrievals."""
        client = self._ensure_client()
        min_uses = max(0, int(min_uses))
        threshold = float(score_threshold)
        to_delete: list[str] = []
        offset = 0
        limit = 1000
        while True:
            batch = client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["memory_id", "metadata"],
                limit=limit,
                offset=offset,
            )
            if not batch:
                break
            for entity in batch:
                mid = entity.get("memory_id")
                if not mid:
                    continue
                meta = entity.get("metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                c_use = int(meta.get(UTILITY_USE_KEY, 0))
                if c_use < min_uses:
                    continue
                c_succ = int(meta.get(UTILITY_SUCC_KEY, 0))
                u_score = read_utility_score_from_metadata(meta, c_use, c_succ)
                if u_score < threshold:
                    to_delete.append(str(mid))
            offset += len(batch)
            if len(batch) < limit:
                break

        if not to_delete:
            return 0

        deleted = 0
        chunk_size = 256
        for start in range(0, len(to_delete), chunk_size):
            chunk = to_delete[start : start + chunk_size]
            expr = ", ".join(f'"{memory_id}"' for memory_id in chunk)
            client.delete(collection_name=self.collection_name, filter=f"memory_id in [{expr}]")
            deleted += len(chunk)
        self.logger.info(
            "prune_low_utility_memories deleted=%s threshold=%.4f min_uses=%s collection=%s",
            deleted,
            threshold,
            min_uses,
            self.collection_name,
        )
        return deleted

    def clear_all_records(self) -> None:
        """Drop and recreate an empty collection (same schema)."""
        client = self._ensure_client()
        collections = set(client.list_collections())
        if self.collection_name in collections:
            client.drop_collection(self.collection_name)
        self._create_collection()
        self.logger.info("clear_all_records recreated collection=%s", self.collection_name)

    def count_records(self) -> int:
        try:
            client = self._ensure_client()
            if self.collection_name not in set(client.list_collections()):
                return 0
            stats = client.get_collection_stats(collection_name=self.collection_name)
            if isinstance(stats, dict):
                for key in ("row_count", "num_entities", "entity_count"):
                    if key in stats:
                        return int(stats[key])
        except Exception as exc:
            self.logger.warning("count_records failed collection=%s err=%s", self.collection_name, exc)
        return -1

    def close(self) -> None:
        if self._client is None:
            return
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
        self._client = None
        self.logger.info("closed client db_path=%s", self.db_path)

    @property
    def _vector_field_name(self) -> str:
        base_name = self.retrieve_key.replace("_text", "")
        return f"{base_name}_vector"

    def _ensure_client(self, force_new: bool = False):
        if self._client is not None and not force_new:
            return self._client

        from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

        self._milvus_client_cls = MilvusClient
        self._collection_schema_cls = CollectionSchema
        self._field_schema_cls = FieldSchema
        self._data_type = DataType
        self._client = MilvusClient(self.db_path)
        return self._client

    def _create_collection(self) -> None:
        client = self._ensure_client()
        vector_field_name = self._vector_field_name
        FieldSchema = self._field_schema_cls
        DataType = self._data_type
        CollectionSchema = self._collection_schema_cls

        fields = [
            FieldSchema(name="memory_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="task_name", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="item_id", dtype=DataType.INT64),
            FieldSchema(name="source_episode_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="source_step", dtype=DataType.INT64),
            FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="state_text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="action_text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="memory_text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="reward", dtype=DataType.FLOAT),
            FieldSchema(name="success", dtype=DataType.BOOL),
            FieldSchema(name="created_step", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="retrieval_count", dtype=DataType.INT64),
            FieldSchema(name="last_used_step", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="value", dtype=DataType.FLOAT),
            FieldSchema(name="value_source", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="value_update_step", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields=fields, description="External memory records for naive RAG retrieval")
        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            dimension=self.embedding_dim,
            auto_id=False,
        )
        index_params = self._milvus_client_cls.prepare_index_params()
        index_params.add_index(field_name=vector_field_name, index_type="FLAT", metric_type="COSINE")
        client.create_index(collection_name=self.collection_name, index_params=index_params)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text if text else " " for text in texts]
        return self.embedding_provider.get_embeddings(cleaned)

    def _build_logger(self) -> logging.Logger:
        logger_name = f"memadaptor.milvus_store.{os.path.abspath(self.store_dir)}"
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        return logger

    def _insert_entities_in_batches(self, entities: list[dict]) -> int:
        if not entities:
            return 0
        client = self._ensure_client()
        inserted = 0
        for start in range(0, len(entities), self.rebuild_insert_batch_size):
            chunk = entities[start : start + self.rebuild_insert_batch_size]
            client.insert(collection_name=self.collection_name, data=chunk)
            inserted += len(chunk)
        return inserted

    def _iter_collection_records_from_client(
        self,
        client,
        collection_name: str,
        include_vectors: bool = True,
        vector_field_name: str | None = None,
    ) -> list[dict]:
        vector_field_name = vector_field_name or self._vector_field_name
        output_fields = [
            "memory_id",
            "task_name",
            "item_id",
            "source_episode_id",
            "source_step",
            "state_text",
            "action_text",
            "memory_text",
            "reward",
            "success",
            "created_step",
            "retrieval_count",
            "last_used_step",
            "metadata",
            "value",
            "value_source",
            "value_update_step",
            "created_at",
        ]
        if include_vectors:
            output_fields.append(vector_field_name)

        records: list[dict] = []
        offset = 0
        limit = 1000
        while True:
            batch = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=output_fields,
                limit=limit,
                offset=offset,
            )
            if not batch:
                break
            records.extend(batch)
            offset += len(batch)
            if len(batch) < limit:
                break
        return records

    def _rebuild_from_db(self, source_db_path: str, source_collection_name: str | None = None) -> int:
        """Copy vectors from another Milvus-Lite file into the current ``milvus_memory.db``."""
        client = self._ensure_client()
        source_client = self._milvus_client_cls(db_path=source_db_path)
        source_collection = source_collection_name or self.collection_name
        collections = set(source_client.list_collections())
        if source_collection not in collections:
            raise ValueError(f"Collection {source_collection} not found in source db {source_db_path}")

        if self.collection_name in set(client.list_collections()):
            client.drop_collection(self.collection_name)
        self._create_collection()

        rows = self._iter_collection_records_from_client(
            client=source_client,
            collection_name=source_collection,
            include_vectors=True,
        )
        inserted = self._insert_entities_in_batches(rows)
        close_fn = getattr(source_client, "close", None)
        if callable(close_fn):
            close_fn()
        self.logger.info(
            "rebuild_from_db complete target_db=%s target_collection=%s source_db=%s source_collection=%s rows_inserted=%s",
            self.db_path,
            self.collection_name,
            source_db_path,
            source_collection,
            inserted,
        )
        return inserted

    def _rebuild_from_jsonl(self, source_jsonl_path: str) -> int:
        """Replace the target collection in the **current** ``milvus_memory.db`` with data from JSONL.

        Drops the existing collection if present, recreates the schema, embeds text fields in batches,
        and inserts. This overwrites in-place vectors for ``self.collection_name``; it does not create
        a new filename.
        """
        client = self._ensure_client()
        if self.collection_name in set(client.list_collections()):
            client.drop_collection(self.collection_name)
            self.logger.info(
                "_rebuild_from_jsonl dropped existing collection=%s before import",
                self.collection_name,
            )
        self._create_collection()
        self.logger.info(
            "_rebuild_from_jsonl created fresh collection=%s reading=%s",
            self.collection_name,
            source_jsonl_path,
        )

        records: list[MemoryRecord] = []
        with open(source_jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "memory_id" not in payload:
                    continue
                records.append(MemoryRecord.from_dict(payload))

        inserted = 0
        for start in range(0, len(records), self.rebuild_embedding_batch_size):
            chunk = records[start : start + self.rebuild_embedding_batch_size]
            self.add_records(chunk)
            inserted += len(chunk)
        self.logger.info(
            "rebuild_from_jsonl complete collection=%s target_db=%s source=%s records_inserted=%s",
            self.collection_name,
            self.db_path,
            source_jsonl_path,
            inserted,
        )
        return inserted

    def _entity_from_row(self, row: dict) -> dict:
        vector_field_name = self._vector_field_name
        metadata = row.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata, ensure_ascii=False)
        return {
            "memory_id": _safe_str(row.get("memory_id"), 128),
            "task_name": _safe_str(row.get("task_name"), 128),
            "item_id": int(row.get("item_id", 0)),
            "source_episode_id": _safe_str(row.get("source_episode_id"), 128),
            "source_step": int(row.get("source_step", 0)),
            vector_field_name: row.get(vector_field_name, [0.0] * self.embedding_dim),
            "state_text": _safe_str(row.get("state_text"), 4096),
            "action_text": _safe_str(row.get("action_text"), 4096),
            "memory_text": _safe_str(row.get("memory_text"), 8192),
            "reward": _safe_float(row.get("reward")),
            "success": bool(row.get("success")),
            "created_step": _safe_str(row.get("created_step"), 64),
            "retrieval_count": int(row.get("retrieval_count", 0)),
            "last_used_step": _safe_str(row.get("last_used_step"), 64),
            "metadata": _safe_str(metadata, 8192),
            "value": _safe_float(row.get("value")),
            "value_source": _safe_str(row.get("value_source"), 64),
            "value_update_step": _safe_str(row.get("value_update_step"), 64),
            "created_at": _safe_str(row.get("created_at"), 64),
        }
