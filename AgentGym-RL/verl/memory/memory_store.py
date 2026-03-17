from __future__ import annotations

import json
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
    "load_only",
}


class BaseMemoryStore:
    def initialize(self, mode: str) -> None:
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
        retrieve_key: str = "state_text",
        can_write: bool = True,
        bootstrap_path: str | None = None,
        bootstrap_collection_name: str | None = None,
    ) -> None:
        self.task_name = task_name
        self.store_dir = store_dir
        self.db_path = db_path or os.path.join(store_dir, "milvus_memory.db")
        self.collection_name = _normalize_collection_name(task_name=task_name, collection_name=collection_name)
        self.only_successful = only_successful
        self.top_k = top_k
        self.min_score = min_score
        self.timeout = timeout
        self.embedding_dim = embedding_dim
        self.can_write = can_write
        self.bootstrap_path = bootstrap_path
        self.bootstrap_collection_name = bootstrap_collection_name
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

    def initialize(self, mode: str) -> None:
        if not self.can_write:
            return

        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in SUPPORTED_MILVUS_INIT_MODES:
            raise ValueError(
                f"Unsupported Milvus memory mode: {mode}. "
                f"Expected one of {sorted(SUPPORTED_MILVUS_INIT_MODES)}."
            )

        collections = set(self.client.list_collections())
        if normalized_mode == "recreate" and self.collection_name in collections:
            self.client.drop_collection(self.collection_name)
            collections.remove(self.collection_name)

        if normalized_mode == "load_only":
            if self.collection_name not in collections:
                raise ValueError(
                    f"Milvus collection {self.collection_name} does not exist in {self.db_path} "
                    f"but mode=load_only requires an existing collection."
                )
            return

        if self.collection_name not in collections:
            self._create_collection()

        if self.bootstrap_path and (
            normalized_mode == "recreate" or self._collection_count(self.collection_name) == 0
        ):
            inserted = self.rebuild_from_path(
                source_path=self.bootstrap_path,
                source_collection_name=self.bootstrap_collection_name,
            )
            print(f"Bootstrapped {inserted} memory records from {self.bootstrap_path}")

    def sync(self, current_step: int | None) -> None:
        pass

    def add_records(self, records: Iterable[MemoryRecord]) -> None:
        if not self.can_write:
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

        now = int(time.time())
        entities = []
        for index, record in enumerate(buffered):
            entities.append(
                {
                    "memory_id": _safe_str(record.memory_id, 128),
                    "task_name": _safe_str(record.task_name, 128),
                    "item_id": int(record.item_id),
                    "source_episode_id": _safe_str(record.source_episode_id, 128),
                    "source_step": int(record.source_step),
                    "state_vector": state_vectors[index],
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
                    "created_at": now,
                }
            )

        try:
            self.client.insert(collection_name=self.collection_name, data=entities)
        except Exception as exc:
            print(f"Failed to insert memory records into Milvus: {exc}")

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
            print(f"Failed to search memories from Milvus: {exc}")
            return []

        hits = results[0] if results else []
        retrieved: list[RetrievedMemory] = []
        for hit in hits:
            entity = hit.get("entity", {})
            score = _safe_float(hit.get("distance", hit.get("score", 0.0)))
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
        return retrieved

    def export_jsonl(self, output_path: str, include_vectors: bool = True) -> int:
        records = self._iter_collection_records(self.collection_name, include_vectors=include_vectors)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        exported = 0
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1
        return exported

    def rebuild_from_path(self, source_path: str, source_collection_name: str | None = None) -> int:
        source_path = os.path.abspath(source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Memory rebuild source does not exist: {source_path}")
        if source_path.endswith(".jsonl"):
            return self._rebuild_from_jsonl(source_path)
        if source_path.endswith(".db"):
            return self._rebuild_from_db(source_path, source_collection_name=source_collection_name)
        raise ValueError(f"Unsupported rebuild source: {source_path}")

    def _create_collection(self) -> None:
        fields = [
            FieldSchema(name="memory_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="task_name", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="item_id", dtype=DataType.INT64),
            FieldSchema(name="source_episode_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="source_step", dtype=DataType.INT64),
            FieldSchema(name="state_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
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
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, description="Memory records for rollout retrieval")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            dimension=self.embedding_dim,
            auto_id=False,
        )
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="state_vector", index_type="FLAT", metric_type="COSINE")
        self.client.create_index(collection_name=self.collection_name, index_params=index_params)

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
            output_fields.append("state_vector")

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
            return 0
        entities = [self._entity_from_row(row) for row in rows]
        self.client.insert(collection_name=self.collection_name, data=entities)
        return len(entities)

    def _rebuild_from_jsonl(self, source_jsonl_path: str) -> int:
        if self.collection_name in set(self.client.list_collections()):
            self.client.drop_collection(self.collection_name)
        self._create_collection()

        rows: list[dict] = []
        with open(source_jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                rows.append(json.loads(payload))

        if not rows:
            return 0

        entities = self._entities_from_payload_rows(rows)
        self.client.insert(collection_name=self.collection_name, data=entities)
        return len(entities)

    def _entities_from_payload_rows(self, rows: list[dict]) -> list[dict]:
        missing_vector_rows = [row for row in rows if "state_vector" not in row]
        embedded_vectors: list[list[float]] = []
        if missing_vector_rows:
            embedded_vectors = self.embedding_provider.get_embeddings(
                [str(row.get(self.retrieve_key, "state_text")) for row in missing_vector_rows]
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

    def _entity_from_row(self, row: dict) -> dict:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, str):
            metadata = _json_dumps(metadata)
        return {
            "memory_id": _safe_str(row.get("memory_id"), 128),
            "task_name": _safe_str(row.get("task_name"), 128),
            "item_id": int(row.get("item_id", 0)),
            "source_episode_id": _safe_str(row.get("source_episode_id"), 128),
            "source_step": int(row.get("source_step", 0)),
            "state_vector": row.get("state_vector", [0.0] * self.embedding_dim),
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
            "created_at": int(row.get("created_at", int(time.time()))),
        }


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
        retrieve_key: str = "state_text",
        can_write: bool = True,
        bootstrap_path: str | None = None,
        bootstrap_collection_name: str | None = None,
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
            bootstrap_path=bootstrap_path,
            bootstrap_collection_name=bootstrap_collection_name,
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
