import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from verl.memory.memory_store import SUPPORTED_MILVUS_INIT_MODES, VectorMemoryStore


def build_store(args: argparse.Namespace) -> VectorMemoryStore:
    return VectorMemoryStore(
        backend="milvus",
        task_name=args.task_name,
        store_dir=args.store_dir,
        db_path=args.db_path,
        collection_name=args.collection_name,
        timeout=args.timeout,
        only_successful=args.only_successful,
        top_k=args.top_k,
        min_score=args.min_score,
        embedding_api_url=args.embedding_api_url,
        embedding_api_key=args.embedding_api_key,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        retrieve_key=args.retrieve_key,
        can_write=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or rebuild the local Milvus memory database.")
    parser.add_argument("--action", default="export", choices=["export", "rebuild"])
    parser.add_argument(
        "--mode",
        default="init_if_missing",
        choices=sorted(SUPPORTED_MILVUS_INIT_MODES),
        help=(
            "Milvus init mode: "
            "init_if_missing=create collection when missing and optionally bootstrap if empty; "
            "recreate=drop and rebuild collection from scratch; "
            "load_only=require an existing collection and never create/rebuild it."
        ),
    )
    parser.add_argument("--task-name", default="alfworld")
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--db-path", default="data/MemAdaptor/alfworld_memory_test.db")
    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--source-collection-name", default=None)
    parser.add_argument("--output-path", default="data/MemAdaptor/alfworld_memory_export_test.jsonl")
    parser.add_argument("--include-vectors", action="store_true")
    parser.add_argument("--embedding-api-url", default="http://10.140.37.68:8081/v1")
    parser.add_argument("--embedding-api-key", default="empty")
    parser.add_argument("--embedding-model", default="bge_m3")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--retrieve-key", default="state_text")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--only-successful", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = build_store(args)
    store.initialize(mode=args.mode)

    if args.action == "export":
        if not args.output_path:
            raise ValueError("--output-path is required when action=export")
        exported = store.export_jsonl(output_path=args.output_path, include_vectors=args.include_vectors)
        print(f"Exported {exported} records to {args.output_path}")
        return

    if not args.source_path:
        raise ValueError("--source-path is required when action=rebuild")
    rebuilt = store.rebuild_from_path(
        source_path=args.source_path,
        source_collection_name=args.source_collection_name,
    )
    print(f"Rebuilt {rebuilt} records into {args.db_path}")


if __name__ == "__main__":
    main()
