#!/usr/bin/env python3
"""Export a Milvus-Lite ``milvus_memory.db`` to JSONL and rebuild under a new store / retrieve_key.

Typical use: copy a ``memory_text``-indexed VDB to a new directory with ``state_text`` vectors
(for ``retrieval_mode=fixed`` + ``retrieve_key=state_text``).

Example::

  python3 scripts/memory_db_export_rebuild.py \\
    --source_store_dir \\
      data/MemAdaptor/exp_results/alfworld/train_adaptor-same-7B-cold_start_20260519_epoch1-1-with_agentic_memory-retrieve_memory_text-self_distill/memory_vdb \\
    --source_retrieve_key memory_text \\
    --target_store_dir \\
      data/MemAdaptor/exp_results/alfworld/qwen2.5-7b-fixed-vdb/memory_vdb \\
    --target_retrieve_key state_text \\
    --task_name alfworld/AlfredTWEnv \\
    --embedding_api_url http://10.140.37.57:8081/v1

Only export (keep JSONL for inspection)::

  python3 scripts/memory_db_export_rebuild.py --action export_only ... --jsonl_path /tmp/mem_export.jsonl

Rebuild from an existing JSONL::

  python3 scripts/memory_db_export_rebuild.py --action rebuild_only \\
    --jsonl_path /tmp/mem_export.jsonl --target_store_dir ... --target_retrieve_key state_text

On CentOS7 login nodes (old GLIBC), Milvus Lite cannot open ``.db`` directly. Use the Apptainer wrapper::

  bash scripts/run_memory_db_export_rebuild.sh --action export_only ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_system.memory import MemoryStoreDispatcher
from agent_system.memory.types import MemoryRecord

_MEMORY_RECORD_FIELDS = frozenset(MemoryRecord.__dataclass_fields__.keys())
_VECTOR_SUFFIXES = ("_vector",)


def _resolve_store_dir(source: str) -> str:
    """Accept either a memory_vdb directory or a direct ``milvus_memory.db`` path."""
    path = os.path.abspath(source)
    if os.path.isdir(path):
        return path
    if os.path.isfile(path) and path.endswith(".db"):
        return os.path.dirname(path)
    raise FileNotFoundError(f"Source store dir or .db file not found: {source}")


def _default_jsonl_path(source_store_dir: str, target_retrieve_key: str) -> str:
    base = os.path.basename(os.path.normpath(source_store_dir))
    parent = os.path.dirname(os.path.normpath(source_store_dir))
    return os.path.join(parent, f"{base}_export_{target_retrieve_key}.jsonl")


def _parse_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"raw_metadata": raw}
        except json.JSONDecodeError:
            return {"raw_metadata": raw}
    return {}


def _normalize_export_row(row: dict[str, Any]) -> dict[str, Any]:
    """Strip Milvus vector columns; keep fields compatible with ``MemoryRecord``."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if any(key.endswith(suffix) for suffix in _VECTOR_SUFFIXES):
            continue
        if key in _MEMORY_RECORD_FIELDS:
            out[key] = value
    if "metadata" in out:
        out["metadata"] = _parse_metadata(out["metadata"])
    return out


def build_store(
    *,
    store_dir: str,
    task_name: str,
    retrieve_key: str,
    collection_name: str | None,
    embedding_api_url: str | None,
    embedding_api_key: str,
    embedding_model: str,
    embedding_dim: int,
    timeout: int,
    rebuild_insert_batch_size: int,
    rebuild_embedding_batch_size: int,
) -> MemoryStoreDispatcher:
    return MemoryStoreDispatcher(
        backend="milvus",
        task_name=task_name,
        store_dir=store_dir,
        collection_name=collection_name,
        timeout=timeout,
        retrieve_key=retrieve_key,
        embedding_api_url=embedding_api_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        rebuild_insert_batch_size=rebuild_insert_batch_size,
        rebuild_embedding_batch_size=rebuild_embedding_batch_size,
    )


def export_db_to_jsonl(
    *,
    source_store_dir: str,
    source_retrieve_key: str,
    jsonl_path: str,
    task_name: str,
    collection_name: str | None,
    embedding_api_url: str | None,
    embedding_api_key: str,
    embedding_model: str,
    embedding_dim: int,
    timeout: int,
) -> int:
    """Open source ``milvus_memory.db`` and write normalized JSONL (no vectors)."""
    store = build_store(
        store_dir=source_store_dir,
        task_name=task_name,
        retrieve_key=source_retrieve_key,
        collection_name=collection_name,
        embedding_api_url=embedding_api_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        timeout=timeout,
        rebuild_insert_batch_size=1000,
        rebuild_embedding_batch_size=256,
    )
    try:
        store.initialize(mode="init_if_missing", clean_before_init=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            store.export_jsonl(output_path=tmp_path, include_vectors=False)
        except Exception:
            os.remove(tmp_path)
            raise

        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)
        exported = 0
        with open(tmp_path, "r", encoding="utf-8") as src, open(jsonl_path, "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                row = _normalize_export_row(json.loads(line))
                if not row.get("memory_id"):
                    continue
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                exported += 1
        os.remove(tmp_path)
        print(f"Exported {exported} records -> {jsonl_path}")
        return exported
    finally:
        store.close()


def rebuild_jsonl_to_db(
    *,
    jsonl_path: str,
    target_store_dir: str,
    target_retrieve_key: str,
    task_name: str,
    collection_name: str | None,
    embedding_api_url: str | None,
    embedding_api_key: str,
    embedding_model: str,
    embedding_dim: int,
    timeout: int,
    rebuild_insert_batch_size: int,
    rebuild_embedding_batch_size: int,
    clean_before_init: bool,
) -> int:
    """Rebuild target ``milvus_memory.db``; re-embed using ``target_retrieve_key``."""
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    records: list[MemoryRecord] = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = _normalize_export_row(json.loads(line))
            if not row.get("memory_id"):
                continue
            records.append(MemoryRecord.from_dict(row))

    store = build_store(
        store_dir=target_store_dir,
        task_name=task_name,
        retrieve_key=target_retrieve_key,
        collection_name=collection_name,
        embedding_api_url=embedding_api_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        timeout=timeout,
        rebuild_insert_batch_size=rebuild_insert_batch_size,
        rebuild_embedding_batch_size=rebuild_embedding_batch_size,
    )
    try:
        os.makedirs(target_store_dir, exist_ok=True)
        store.initialize(mode="init_if_missing", clean_before_init=clean_before_init)
        inserted = 0
        batch = max(1, rebuild_embedding_batch_size)
        for start in range(0, len(records), batch):
            chunk = records[start : start + batch]
            store.add_records(chunk)
            inserted += len(chunk)

        db_path = os.path.join(target_store_dir, "milvus_memory.db")
        vector_field = f"{target_retrieve_key.replace('_text', '')}_vector"
        print(
            json.dumps(
                {
                    "inserted": inserted,
                    "target_store_dir": target_store_dir,
                    "db_path": db_path,
                    "retrieve_key": target_retrieve_key,
                    "vector_field": vector_field,
                    "jsonl_path": jsonl_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return inserted
    finally:
        store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Milvus-Lite .db to JSONL and rebuild with a (possibly new) retrieve_key.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--action",
        default="export_and_rebuild",
        choices=["export_and_rebuild", "export_only", "rebuild_only"],
        help="export_and_rebuild: export source db then rebuild target (default).",
    )
    parser.add_argument(
        "--source_store_dir",
        default="",
        help="Source memory_vdb directory, or path to milvus_memory.db.",
    )
    parser.add_argument(
        "--source_retrieve_key",
        default="memory_text",
        help="retrieve_key used when the source .db was built (for reading export).",
    )
    parser.add_argument(
        "--target_store_dir",
        default="",
        help="Output memory_vdb directory for the rebuilt database.",
    )
    parser.add_argument(
        "--target_retrieve_key",
        default="state_text",
        help="retrieve_key for the rebuilt database (vectors re-embedded from this field).",
    )
    parser.add_argument(
        "--jsonl_path",
        default="",
        help="JSONL path. Default: <source_store_dir>_export_<target_retrieve_key>.jsonl",
    )
    parser.add_argument("--task_name", default="alfworld/AlfredTWEnv")
    parser.add_argument("--collection_name", default=None)
    parser.add_argument("--embedding_api_url", default="http://10.140.37.57:8081/v1")
    parser.add_argument("--embedding_api_key", default="")
    parser.add_argument("--embedding_model", default="bge_m3")
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--rebuild_insert_batch_size", type=int, default=1000)
    parser.add_argument("--rebuild_embedding_batch_size", type=int, default=256)
    parser.add_argument(
        "--clean_before_init",
        action="store_true",
        default=True,
        help="Remove existing milvus_memory.db* in target_store_dir before rebuild (default: true).",
    )
    parser.add_argument(
        "--no_clean_before_init",
        action="store_false",
        dest="clean_before_init",
        help="Keep existing target db files (not recommended when changing retrieve_key).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = args.jsonl_path

    if args.action in ("export_and_rebuild", "export_only"):
        if not args.source_store_dir:
            raise ValueError("--source_store_dir is required for export")
        source_store_dir = _resolve_store_dir(args.source_store_dir)
        jsonl_path = jsonl_path or _default_jsonl_path(source_store_dir, args.target_retrieve_key)
        export_db_to_jsonl(
            source_store_dir=source_store_dir,
            source_retrieve_key=args.source_retrieve_key,
            jsonl_path=jsonl_path,
            task_name=args.task_name,
            collection_name=args.collection_name,
            embedding_api_url=args.embedding_api_url or None,
            embedding_api_key=args.embedding_api_key,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            timeout=args.timeout,
        )

    if args.action in ("export_and_rebuild", "rebuild_only"):
        if not args.target_store_dir:
            raise ValueError("--target_store_dir is required for rebuild")
        if not jsonl_path:
            raise ValueError("--jsonl_path is required when action=rebuild_only")
        rebuild_jsonl_to_db(
            jsonl_path=jsonl_path,
            target_store_dir=os.path.abspath(args.target_store_dir),
            target_retrieve_key=args.target_retrieve_key,
            task_name=args.task_name,
            collection_name=args.collection_name,
            embedding_api_url=args.embedding_api_url or None,
            embedding_api_key=args.embedding_api_key,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            timeout=args.timeout,
            rebuild_insert_batch_size=args.rebuild_insert_batch_size,
            rebuild_embedding_batch_size=args.rebuild_embedding_batch_size,
            clean_before_init=args.clean_before_init,
        )


if __name__ == "__main__":
    main()

