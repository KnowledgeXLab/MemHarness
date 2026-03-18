from __future__ import annotations

import atexit
import json
import logging
import os
import re
import time
from typing import Iterable
import httpx
from openai import OpenAI
import requests
from pymilvus import MilvusClient, CollectionSchema, DataType, FieldSchema

from verl.memory.types import MemoryRecord, RetrievedMemory


def _json_dumps(payload: object) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return "{}"


def _safe_str(value: object, max_length: int) -> str:
    text = "" if value is None else str(value)
    if max_length > 0 and len(text) > max_length:
        print(f"Truncating string to {max_length} characters")
        return text[:max_length]
    return text


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        print(f"Failed to convert value to float: {value}")
        return default
    if number != number or number in (float("inf"), float("-inf")):
        print(f"Invalid float value: {number}")
        return default
    return number


def _normalize_collection_name(task_name: str, collection_name: str | None) -> str:
    if collection_name:
        return collection_name
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", task_name or "memory")
    return f"agent_memories_{normalized}".strip("_")


SUPPORTED_MILVUS_INIT_MODES = {
    "init_if_missing",
    "recreate",
}


class BaseMemoryStore:
    def initialize(self, mode: str, clean_before_init: bool = False) -> None:
        raise NotImplementedError

    def sync(self, current_step: int | None) -> None:
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

    def close(self) -> None:
        raise NotImplementedError

    def delete_records(self, memory_ids: Iterable[str]) -> int:
        raise NotImplementedError


# TODO: Implement this class
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
            raise ValueError("memory.vdb_base_url must be set when using the HTTP memory backend.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.only_successful = only_successful
        self.top_k = top_k
        self.min_score = min_score

    def initialize(self, mode: str) -> None:
        del mode

    def sync(self, current_step: int | None) -> None:
        del current_step

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        buffered = list(records)
        if not buffered:
            return
        
        url = f"{self.base_url}/memories/batch"
        payload = {"items": [record.to_dict() for record in buffered]}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Failed to batch add memory records to VDB: {exc}")

    def retrieve(self, query_text: str) -> list[RetrievedMemory]:
        if not query_text:
            return []
            
        url = f"{self.base_url}/memories/search"
        payload = {
            "query": query_text,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "only_successful": self.only_successful,
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return _deserialize_retrieved_items(response.json())
        except requests.RequestException as exc:
            print(f"Failed to search memories from VDB: {exc}")
            return []

    def export_jsonl(self, output_path: str, include_vectors: bool = True) -> int:
        del output_path, include_vectors
        raise NotImplementedError("HTTP memory backend does not support local export.")

    def rebuild_from_path(self, source_path: str, source_collection_name: str | None = None) -> int:
        del source_path, source_collection_name
        raise NotImplementedError("HTTP memory backend does not support local rebuild.")

    def update_records(self, updates: Iterable[dict]) -> int:
        del updates
        raise NotImplementedError("HTTP memory backend does not support local update_records.")

    def close(self) -> None:
        return

    def delete_records(self, memory_ids: Iterable[str]) -> int:
        del memory_ids
        raise NotImplementedError("Delete VDB interface is reserved but not implemented.")


class RemoteEmbeddingProvider:
    def __init__(
        self,
        api_url: str | None,
        api_key: str = "empty",
        model_name: str = "bge_m3",
        embedding_dim: int = 1024,
        timeout: int = 30,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.timeout = timeout
        self.client = None

        if not api_url:
            return

        try:
            http_client = httpx.Client(verify=False, trust_env=False, timeout=float(timeout))
            self.client = OpenAI(api_key=api_key, base_url=api_url, http_client=http_client)
        except Exception as exc:
            print(f"Failed to initialize embedding client, fallback to zero vectors: {exc}")
            self.client = None

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.client is None:
            raise ValueError("Embedding client is not initialized")

        try:
            embedding_obj = self.client.embeddings.create(input=texts, model=self.model_name)
            embeddings = [list(item.embedding) for item in embedding_obj.data]
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding service returned {len(embeddings)} vectors for {len(texts)} inputs. Please check the embedding service is working correctly."
                )
            return embeddings
        except Exception as exc:
            print(f"Failed to fetch embeddings, fallback to zero vectors: {exc}")
            return [[0.0] * self.embedding_dim for _ in texts]


class MilvusMemoryStore(BaseMemoryStore):
    def __init__(
        self,
        task_name: str,
        store_dir: str,
        db_path: str | None = None,
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
        can_write: bool = True,
        rebuild_insert_batch_size: int = 100,
        rebuild_embedding_batch_size: int = 64,
    ) -> None:
        self.task_name = task_name
        self.store_dir = store_dir
        self.db_path = db_path or os.path.join(store_dir, "milvus_memory.db")
        self.collection_name = _normalize_collection_name(
            task_name=task_name, collection_name=collection_name, use_timestamp=(collection_name is None)
        )
        self.only_successful = only_successful
        self.top_k = top_k
        self.min_score = min_score
        self.timeout = timeout
        self.embedding_dim = embedding_dim
        self.can_write = can_write
        self.rebuild_insert_batch_size = max(1, int(rebuild_insert_batch_size))
        self.rebuild_embedding_batch_size = max(1, int(rebuild_embedding_batch_size))
        self.embedding_provider = RemoteEmbeddingProvider(
            api_url=embedding_api_url,
            api_key=embedding_api_key,
            model_name=embedding_model,
            embedding_dim=embedding_dim,
            timeout=timeout,
        )
        self.retrieve_key = retrieve_key

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.client = MilvusClient(self.db_path)
        self.log_path = os.path.join(
            os.path.dirname(self.db_path),
            f"{os.path.splitext(os.path.basename(self.db_path))[0]}.log",
        )
        self.logger = self._build_logger()
        self.logger.info(
            "Initialized MilvusMemoryStore db_path=%s collection_name=%s can_write=%s",
            self.db_path,
            self.collection_name,
            self.can_write,
        )
        self.logger.info(
            "Rebuild batch sizes embedding=%s insert=%s",
            self.rebuild_embedding_batch_size,
            self.rebuild_insert_batch_size,
        )
        atexit.register(self.close)

    def initialize(self, mode: str) -> None:
        if not self.can_write:
            self.logger.info("Skip initialize because can_write=False")
            return

        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in SUPPORTED_MILVUS_INIT_MODES:
            raise ValueError(
                f"Unsupported Milvus memory mode: {mode}. "
                f"Expected one of {sorted(SUPPORTED_MILVUS_INIT_MODES)}."
            )

        collections = set(self.client.list_collections())
        self.logger.info("Initialize collection with mode=%s existing_collections=%s", normalized_mode, sorted(collections))
        if normalized_mode == "recreate" and self.collection_name in collections:
            self.client.drop_collection(self.collection_name)
            collections.remove(self.collection_name)
            self.logger.info("Dropped existing collection %s before recreate", self.collection_name)

        if self.collection_name not in collections:
            self._create_collection()
            self.logger.info("Created collection %s", self.collection_name)

    def sync(self, current_step: int | None) -> None:
        self.logger.info("Sync called current_step=%s", current_step)

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        if not self.can_write:
            self.logger.info("Skip add_records because can_write=False")
            return

        buffered = list(records)
        if not buffered:
            return

        state_vectors = self.embedding_provider.get_embeddings(
            [getattr(record, self.retrieve_key, "") for record in buffered]
        )
        if len(state_vectors) != len(buffered):
            print(f"Embedding count mismatch when inserting memory records, skipping insert. Expected {len(buffered)} vectors, got {len(state_vectors)}.")
            return

        vector_field_name = self._vector_field_name
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        entities = []
        for index, record in enumerate(buffered):
            entities.append(
                {
                    "memory_id": _safe_str(record.memory_id, 128),
                    "task_name": _safe_str(record.task_name, 128),
                    "item_id": int(record.item_id),
                    "source_episode_id": _safe_str(record.source_episode_id, 128),
                    "source_step": int(record.source_step),
                    vector_field_name: state_vectors[index],
                    "state_text": _safe_str(record.state_text, 4096),
                    "action_text": _safe_str(record.action_text, 4096),
                    "memory_text": _safe_str(record.memory_text, 8192),
                    "reward": _safe_float(record.reward),
                    "success": bool(record.success),
                    "created_step": _safe_str(record.created_step, 64),
                    "retrieval_count": int(record.retrieval_count),
                    "last_used_step": _safe_str(record.last_used_step, 64),
                    "metadata": _safe_str(_json_dumps(record.metadata), 8192),
                    "value": _safe_float(record.value),
                    "value_source": _safe_str(record.value_source, 64),
                    "value_update_step": _safe_str(record.value_update_step, 64),
                    "created_at": now_ts,
                }
            )

        try:
            inserted = self._insert_entities_in_batches(entities)
            self.logger.info("Inserted %s memory records into %s", inserted, self.collection_name)
        except Exception as exc:
            self.logger.exception("Failed to insert memory records into Milvus: %s", exc)

    def retrieve(self, query_text: str) -> list[RetrievedMemory]:
        if not query_text:
            return []

        query_vector = self.embedding_provider.get_embeddings([query_text])[0]
        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "limit": max(1, self.top_k),
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

        try:
            results = self.client.search(**search_kwargs)
        except Exception as exc:
            self.logger.exception("Failed to search memories from Milvus: %s", exc)
            return []

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
        self.logger.info("Retrieved %s memories for query length=%s", len(retrieved), len(query_text[:50]))
        return retrieved

    def export_jsonl(self, output_path: str, include_vectors: bool = True) -> int:
        records = self._iter_collection_records(self.collection_name, include_vectors=include_vectors)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        exported = 0
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1
        self.logger.info("Exported %s records to %s include_vectors=%s", exported, output_path, include_vectors)
        return exported

    def rebuild_from_path(self, source_path: str, source_collection_name: str | None = None) -> int:
        source_path = os.path.abspath(source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Memory rebuild source does not exist: {source_path}")
        self.logger.info("Start rebuild source=%s target_collection=%s", source_path, self.collection_name)
        try:
            if source_path.endswith(".jsonl"):
                rebuilt = self._rebuild_from_jsonl(source_path)
                self.logger.info("Rebuilt %s records from jsonl source=%s", rebuilt, source_path)
                return rebuilt
            if source_path.endswith(".db"):
                rebuilt = self._rebuild_from_db(source_path, source_collection_name=source_collection_name)
                self.logger.info(
                    "Rebuilt %s records from db source=%s source_collection=%s",
                    rebuilt,
                    source_path,
                    source_collection_name or self.collection_name,
                )
                return rebuilt
            raise ValueError(f"Unsupported rebuild source: {source_path}")
        except Exception as exc:
            self.logger.exception("Failed rebuild from source=%s: %s", source_path, exc)
            raise

    def update_records(self, updates: Iterable[dict]) -> int:
        buffered_updates = [dict(update) for update in updates if update and update.get("memory_id")]
        if not buffered_updates:
            return 0
        if not self.can_write:
            self.logger.info("Skip update_records because can_write=False")
            return 0

        vector_field_name = self._vector_field_name
        updated_entities: list[dict] = []
        updated_ids: list[str] = []
        for update in buffered_updates:
            memory_id = str(update["memory_id"])
            existing_rows = self.client.query(
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
                self.logger.warning("Skip update for missing memory_id=%s", memory_id)
                continue

            row = dict(existing_rows[0])
            retrieval_increment = int(update.get("_increment_retrieval_count", 0))
            if "metadata" in update and isinstance(update["metadata"], dict):
                current_metadata = row.get("metadata", {})
                if isinstance(current_metadata, str):
                    try:
                        current_metadata = json.loads(current_metadata)
                    except json.JSONDecodeError:
                        current_metadata = {}
                if not isinstance(current_metadata, dict):
                    current_metadata = {}
                merged_metadata = dict(current_metadata)
                merged_metadata.update(update["metadata"])
                update = dict(update)
                update["metadata"] = merged_metadata

            vector_field_name = self._vector_field_name
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
                "metadata",
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
                row[vector_field_name] = self.embedding_provider.get_embeddings([str(row[self.retrieve_key])])[0]
            elif vector_field_name in update:
                row[vector_field_name] = update[vector_field_name]

            updated_entities.append(self._entity_from_row(row))
            updated_ids.append(memory_id)

        if not updated_entities:
            return 0

        expr = ", ".join(f'"{memory_id}"' for memory_id in updated_ids)
        self.client.delete(collection_name=self.collection_name, filter=f"memory_id in [{expr}]")
        self._insert_entities_in_batches(updated_entities)
        self.logger.info("Updated %s memory records", len(updated_entities))
        return len(updated_entities)

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self.client = None

        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info("Closed MilvusMemoryStore for db_path=%s", self.db_path)
            for handler in list(logger.handlers):
                try:
                    handler.flush()
                    handler.close()
                finally:
                    logger.removeHandler(handler)
            self.logger = None

    def delete_records(self, memory_ids: Iterable[str]) -> int:
        del memory_ids
        raise NotImplementedError("Delete VDB interface is reserved but not implemented.")

    @property
    def _vector_field_name(self) -> str:
        """根据 retrieve_key 生成向量字段名，如 state_text -> state_vector"""
        base_name = self.retrieve_key.replace("_text", "")
        return f"{base_name}_vector"

    def _create_collection(self) -> None:
        vector_field_name = self._vector_field_name
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
        schema = CollectionSchema(fields=fields, description="Memory records for rollout retrieval")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            dimension=self.embedding_dim,
            auto_id=False,
        )
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name=vector_field_name, index_type="FLAT", metric_type="COSINE")
        self.client.create_index(collection_name=self.collection_name, index_params=index_params)
        self.logger.info("Created collection schema and index for %s (vector_field=%s, retrieve_key=%s)", self.collection_name, vector_field_name, self.retrieve_key)

    def _collection_count(self, collection_name: str) -> int:
        try:
            stats = self.client.get_collection_stats(collection_name=collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def _iter_collection_records(self, collection_name: str, include_vectors: bool = True) -> list[dict]:
        return self._iter_collection_records_from_client(
            client=self.client,
            collection_name=collection_name,
            include_vectors=include_vectors,
        )

    def _iter_collection_records_from_client(
        self,
        client: MilvusClient,
        collection_name: str,
        include_vectors: bool = True,
    ) -> list[dict]:
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
            output_fields.append(self._vector_field_name)

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
        source_client = MilvusClient(db_path=source_db_path)
        source_collection = source_collection_name or self.collection_name
        if source_collection not in set(source_client.list_collections()):
            raise ValueError(f"Collection {source_collection} not found in source db {source_db_path}")

        if self.collection_name in set(self.client.list_collections()):
            self.client.drop_collection(self.collection_name)
        self._create_collection()

        rows = self._iter_collection_records_from_client(
            client=source_client,
            collection_name=source_collection,
            include_vectors=True,
        )
        if not rows:
            self.logger.info("Source db %s collection=%s is empty", source_db_path, source_collection)
            return 0
        self.logger.info("Loaded %s rows from source db %s collection=%s", len(rows), source_db_path, source_collection)
        entities = [self._entity_from_row(row) for row in rows]
        return self._insert_entities_in_batches(entities, progress_label="DB rebuild")

    def _rebuild_from_jsonl(self, source_jsonl_path: str) -> int:
        if self.collection_name in set(self.client.list_collections()):
            self.client.drop_collection(self.collection_name)
        self._create_collection()

        total_rows = self._count_non_empty_lines(source_jsonl_path)
        self.logger.info(
            "Start JSONL rebuild source=%s total_rows=%s embedding_batch_size=%s insert_batch_size=%s",
            source_jsonl_path,
            total_rows,
            self.rebuild_embedding_batch_size,
            self.rebuild_insert_batch_size,
        )
        if total_rows == 0:
            return 0

        rebuilt = 0
        buffered_rows: list[dict] = []
        with open(source_jsonl_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    buffered_rows.append(json.loads(payload))
                except json.JSONDecodeError as exc:
                    self.logger.exception("Invalid jsonl at %s line=%s: %s", source_jsonl_path, line_no, exc)
                    raise

                if len(buffered_rows) >= self.rebuild_insert_batch_size:
                    rebuilt += self._insert_jsonl_rows_chunk(
                        rows=buffered_rows,
                        inserted_so_far=rebuilt,
                        total_rows=total_rows,
                    )
                    buffered_rows = []

        if buffered_rows:
            rebuilt += self._insert_jsonl_rows_chunk(
                rows=buffered_rows,
                inserted_so_far=rebuilt,
                total_rows=total_rows,
            )
        return rebuilt

    def _insert_jsonl_rows_chunk(self, rows: list[dict], inserted_so_far: int, total_rows: int) -> int:
        entities = self._entities_from_payload_rows(rows, progress_label="JSONL rebuild embedding")
        inserted = self._insert_entities_in_batches(entities)
        self.logger.info("JSONL rebuild progress inserted=%s/%s", inserted_so_far + inserted, total_rows)
        return inserted

    def _entities_from_payload_rows(self, rows: list[dict], progress_label: str | None = None) -> list[dict]:
        missing_vector_rows = [row for row in rows if "state_vector" not in row]
        embedded_vectors: list[list[float]] = []
        if missing_vector_rows:
            embedded_vectors = self._get_embeddings_in_batches(
                [str(row.get(self.retrieve_key, "")) for row in missing_vector_rows],
                batch_size=self.rebuild_embedding_batch_size,
                progress_label=progress_label,
            )

        entities: list[dict] = []
        embedded_index = 0
        for row in rows:
            row_copy = dict(row)
            if "state_vector" not in row_copy:
                row_copy["state_vector"] = embedded_vectors[embedded_index]
                embedded_index += 1
            entities.append(self._entity_from_row(row_copy))
        return entities

    def _get_embeddings_in_batches(
        self,
        texts: list[str],
        batch_size: int,
        progress_label: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, max(1, batch_size)):
            batch = texts[start : start + batch_size]
            batch_vectors = self.embedding_provider.get_embeddings(batch)
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    f"Embedding count mismatch during rebuild. Expected {len(batch)} vectors, got {len(batch_vectors)}."
                )
            embeddings.extend(batch_vectors)
            if progress_label:
                self.logger.info("%s progress embedded=%s/%s", progress_label, len(embeddings), total)
        return embeddings

    def _insert_entities_in_batches(self, entities: list[dict], progress_label: str | None = None) -> int:
        if not entities:
            return 0

        total = len(entities)
        inserted = 0
        for start in range(0, total, self.rebuild_insert_batch_size):
            batch = entities[start : start + self.rebuild_insert_batch_size]
            try:
                self.client.insert(collection_name=self.collection_name, data=batch)
            except Exception as exc:
                first_entity = batch[0] if batch else {}
                self.logger.exception(
                    "Failed to insert batch into Milvus (batch_start=%s, batch_size=%s). "
                    "First entity summary: memory_id=%s, task_name=%s, source_episode_id=%s, "
                    "state_text_len=%s, action_text_len=%s, memory_text_len=%s, metadata_len=%s. "
                    "Exception: %s",
                    start,
                    len(batch),
                    first_entity.get("memory_id", "N/A"),
                    first_entity.get("task_name", "N/A"),
                    first_entity.get("source_episode_id", "N/A"),
                    len(first_entity.get("state_text", "")),
                    len(first_entity.get("action_text", "")),
                    len(first_entity.get("memory_text", "")),
                    len(first_entity.get("metadata", "")),
                    exc,
                )
                raise
            inserted += len(batch)
            if progress_label:
                self.logger.info("%s progress inserted=%s/%s", progress_label, inserted, total)
        return inserted

    @staticmethod
    def _count_non_empty_lines(path: str) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    def _entity_from_row(self, row: dict) -> dict:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, str):
            metadata = _json_dumps(metadata)
        vector_field_name = self._vector_field_name
        return {
            "memory_id": _safe_str(row.get("memory_id"), 128),
            "task_name": _safe_str(row.get("task_name"), 128),
            "item_id": int(row.get("item_id", 0)),
            "source_episode_id": _safe_str(row.get("source_episode_id"), 128),
            "source_step": int(row.get("source_step", 0)),
            vector_field_name: row.get(vector_field_name),
            "state_text": _safe_str(row.get("state_text"), 4096),
            "action_text": _safe_str(row.get("action_text"), 4096),
            "memory_text": _safe_str(row.get("memory_text"), 8192),
            "reward": _safe_float(row.get("reward")),
            "success": bool(row.get("success", False)),
            "created_step": _safe_str(row.get("created_step"), 64),
            "retrieval_count": int(row.get("retrieval_count", 0)),
            "last_used_step": _safe_str(row.get("last_used_step"), 64),
            "metadata": _safe_str(metadata, 8192),
            "value": _safe_float(row.get("value")),
            "value_source": _safe_str(row.get("value_source"), 64),
            "value_update_step": _safe_str(row.get("value_update_step"), 64),
            "created_at": _safe_str(row.get("created_at"), 64),
        }

    def _build_logger(self) -> logging.Logger:
        logger_name = f"verl.memory.milvus.{os.path.abspath(self.db_path)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(handler)
        return logger


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
            score=_safe_float(item.get("score", 0.0)),
            state_text=item.get("state_text", ""),
            action_text=item.get("action_text", ""),
            memory_text=item.get("memory_text", ""),
            reward=_safe_float(item.get("reward", 0.0)),
            metadata=metadata,
            value=item.get("value"),
            value_source=item.get("value_source"),
                )
            )
    return retrieved


class VectorMemoryStore(BaseMemoryStore):
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
        db_path: str | None = None,
        collection_name: str | None = None,
        embedding_api_url: str | None = None,
        embedding_api_key: str = "empty",
        embedding_model: str = "bge_m3",
        embedding_dim: int = 1024,
        retrieve_key: str = "memory_text",
        can_write: bool = True,
        rebuild_insert_batch_size: int = 100,
        rebuild_embedding_batch_size: int = 64,
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

        resolved_store_dir = store_dir or os.path.join(os.getcwd(), "memory_store")
        self.backend = MilvusMemoryStore(
            task_name=task_name,
            store_dir=resolved_store_dir,
            db_path=db_path,
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
            can_write=can_write,
            rebuild_insert_batch_size=rebuild_insert_batch_size,
            rebuild_embedding_batch_size=rebuild_embedding_batch_size,
        )

    def initialize(self, mode: str) -> None:
        self.backend.initialize(mode=mode)

    def sync(self, current_step: int | None) -> None:
        self.backend.sync(current_step=current_step)

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        self.backend.add_records(records)

    def retrieve(self, query_text: str) -> list[RetrievedMemory]:
        return self.backend.retrieve(query_text=query_text)

    def export_jsonl(self, output_path: str, include_vectors: bool = True) -> int:
        return self.backend.export_jsonl(output_path=output_path, include_vectors=include_vectors)

    def rebuild_from_path(self, source_path: str, source_collection_name: str | None = None) -> int:
        return self.backend.rebuild_from_path(
            source_path=source_path,
            source_collection_name=source_collection_name,
        )

    def update_records(self, updates: Iterable[dict]) -> int:
        return self.backend.update_records(updates)

    def close(self) -> None:
        self.backend.close()

    def delete_records(self, memory_ids: Iterable[str]) -> int:
        return self.backend.delete_records(memory_ids)
