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
        rebuild_insert_batch_size=args.rebuild_insert_batch_size,
        rebuild_embedding_batch_size=args.rebuild_embedding_batch_size,
        can_write=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or rebuild the local Milvus memory database.")
    parser.add_argument("--action", default="rebuild", choices=["export", "rebuild"])
    parser.add_argument(
        "--mode",
        default="recreate",
        choices=sorted(SUPPORTED_MILVUS_INIT_MODES),
        help=(
            "Milvus init mode: "
            "init_if_missing=use existing collection or create if missing; "
            "recreate=drop and rebuild collection from source (default for rebuild action)."
        ),
    )
    parser.add_argument("--task_name", default="sciworld")
    parser.add_argument("--store_dir", default="data/AgentGym/AgentTraj-L/memadaptor_test")
    parser.add_argument("--db_path", default=None, help="Path to the Milvus database file. Required when action=export.")
    parser.add_argument("--collection_name", default=None)
    parser.add_argument("--source_path", default="data/AgentGym/AgentTraj-L/sciworld_train_memory_records-gpt-5.1.jsonl", help="Path to the source JSONL file. Required when action=rebuild.")
    parser.add_argument("--source_collection_name", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--include_vectors", action="store_true")
    parser.add_argument("--embedding_api_url", default="http://10.140.37.68:8081/v1")
    parser.add_argument("--embedding_api_key", default="empty")
    parser.add_argument("--embedding_model", default="bge_m3")
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--retrieve_key", default="state_text")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--min_score", type=float, default=0.1)
    parser.add_argument("--only_successful", action="store_true")
    parser.add_argument("--rebuild_insert_batch_size", type=int, default=1000)
    parser.add_argument("--rebuild_embedding_batch_size", type=int, default=256)
    parser.add_argument(
        "--clean_before_init",
        action="store_true",
        help="If set, remove existing database files in store_dir before initialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = build_store(args)
    try:
        if args.action == "export":
            store.initialize(mode=args.mode, clean_before_init=args.clean_before_init)
            if not args.output_path:
                raise ValueError("--output_path is required when action=export")
            exported = store.export_jsonl(output_path=args.output_path, include_vectors=args.include_vectors)
            print(f"Exported {exported} records to {args.output_path}")
            return

        if not args.source_path:
            raise ValueError("--source_path is required when action=rebuild")
        
        # For rebuild action, clean_before_init is handled inside rebuild_from_path by dropping collection
        # But we still pass it to initialize for consistency
        store.initialize(mode=args.mode, clean_before_init=args.clean_before_init)
        
        rebuilt = store.rebuild_from_path(
            source_path=args.source_path,
            source_collection_name=args.source_collection_name,
        )
        print(f"Rebuilt {rebuilt} records into {args.db_path}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
