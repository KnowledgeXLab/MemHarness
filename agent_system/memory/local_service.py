from __future__ import annotations

import atexit
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from .milvus_store import MilvusMemoryStore
from .types import MemoryRecord


def _default_memory_dir(memory_config) -> str:
    memory_dir = memory_config.store_dir
    if memory_dir:
        return memory_dir
    return os.path.join(os.getcwd(), "memory_store")


def _resolve_advertised_host(explicit_host: str | None = None) -> str:
    if explicit_host:
        return explicit_host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


class LocalMemoryServerHandle:
    def __init__(self, process: subprocess.Popen, base_url: str, log_path: str) -> None:
        self.process = process
        self.base_url = base_url
        self.log_path = log_path
        self._closed = False
        atexit.register(self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def start_local_memory_server(memory_config, task_name: str):
    if memory_config is None or not memory_config.enabled:
        return None

    backend = str(memory_config.backend).strip().lower()
    base_url = memory_config.vdb_base_url
    auto_start = bool(memory_config.auto_start_server)

    if backend == "http" and base_url:
        return None
    if not auto_start:
        return None
    if backend not in ("", "milvus", "http"):
        return None

    memory_dir = _default_memory_dir(memory_config=memory_config)
    os.makedirs(memory_dir, exist_ok=True)

    bind_host = str(memory_config.server_bind_host)
    advertised_host = _resolve_advertised_host(memory_config.server_advertised_host)
    requested_port = int(memory_config.server_port)
    port = requested_port if requested_port > 0 else _find_free_port(bind_host)
    startup_timeout = max(1, int(memory_config.server_startup_timeout or 60))
    log_path = os.path.join(memory_dir, "memory_server.log")

    command = [
        sys.executable,
        "-m",
        "agent_system.memory.local_service",
        "--host",
        bind_host,
        "--port",
        str(port),
        "--task-name",
        task_name,
        "--store-dir",
        memory_dir,
        "--mode",
        str(memory_config.mode),
        "--timeout",
        str(memory_config.vdb_timeout),
        "--retrieve-key",
        str(memory_config.retrieve_key),
        "--embedding-dim",
        str(memory_config.embedding_dim),
        "--top-k",
        str(memory_config.top_k),
        "--min-score",
        str(memory_config.min_retrieval_score),
    ]

    for flag_name, arg_name in [
        ("collection_name", "--collection-name"),
        ("rebuild_source_path", "--rebuild-source-path"),
        ("rebuild_source_collection_name", "--rebuild-source-collection-name"),
        ("embedding_api_url", "--embedding-api-url"),
        ("embedding_api_key", "--embedding-api-key"),
        ("embedding_model", "--embedding-model"),
    ]:
        value = getattr(memory_config, flag_name)
        if value:
            command.extend([arg_name, str(value)])

    if memory_config.get("only_successful", True):
        command.append("--only-successful")
    if memory_config.get("clean_before_init", False):
        command.append("--clean-before-init")

    for flag_name, arg_name in [
        ("rebuild_insert_batch_size", "--rebuild-insert-batch-size"),
        ("rebuild_embedding_batch_size", "--rebuild-embedding-batch-size"),
    ]:
        value = memory_config.get(flag_name)
        if value is not None:
            command.extend([arg_name, str(value)])

    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    base_url = f"http://{advertised_host}:{port}"
    deadline = time.time() + startup_timeout
    health_url = f"{base_url}/health"
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Local memory server exited early with code {process.returncode}. Check log: {log_path}"
            )
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == HTTPStatus.OK:
                return LocalMemoryServerHandle(process=process, base_url=base_url, log_path=log_path)
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    raise RuntimeError(
        f"Timed out waiting for local memory server at {health_url}. Last error: {last_error}. Log: {log_path}"
    )


class _MemoryServiceHandler(BaseHTTPRequestHandler):
    server_version = "VerlAgentMemoryServer/1.0"

    @property
    def context(self):
        return self.server.context

    def log_message(self, format: str, *args) -> None:
        return

    def _write_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "task_name": self.context.task_name,
                    "collection_name": self.context.store.collection_name,
                    "store_dir": self.context.store.store_dir,
                },
            )
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {parsed.path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/memories/batch":
                records = [MemoryRecord.from_dict(item) for item in payload.get("items", []) if isinstance(item, dict)]
                with self.context.lock:
                    self.context.store.add_records(records)
                self._write_json(HTTPStatus.OK, {"inserted": len(records)})
                return

            if parsed.path == "/memories/search":
                query_text = str(payload.get("query", "") or "")
                with self.context.lock:
                    old_top_k = self.context.store.top_k
                    old_min_score = self.context.store.min_score
                    old_only_successful = self.context.store.only_successful
                    self.context.store.top_k = int(payload.get("top_k", old_top_k))
                    self.context.store.min_score = float(payload.get("min_score", old_min_score))
                    self.context.store.only_successful = bool(payload.get("only_successful", old_only_successful))
                    try:
                        retrieved = self.context.store.retrieve(query_text=query_text)
                    finally:
                        self.context.store.top_k = old_top_k
                        self.context.store.min_score = old_min_score
                        self.context.store.only_successful = old_only_successful
                self._write_json(HTTPStatus.OK, [item.to_dict() for item in retrieved])
                return

            if parsed.path == "/memories/update":
                updates = payload.get("updates", [])
                if not isinstance(updates, list):
                    raise ValueError("updates must be a list.")
                with self.context.lock:
                    updated = self.context.store.update_records(updates)
                self._write_json(HTTPStatus.OK, {"updated": updated})
                return

            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {parsed.path}"})
        except Exception as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def _build_store_from_args(args: argparse.Namespace) -> MilvusMemoryStore:
    store = MilvusMemoryStore(
        task_name=args.task_name,
        store_dir=args.store_dir,
        collection_name=args.collection_name,
        embedding_api_url=args.embedding_api_url,
        embedding_api_key=args.embedding_api_key,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        timeout=args.timeout,
        only_successful=args.only_successful,
        top_k=args.top_k,
        min_score=args.min_score,
        retrieve_key=args.retrieve_key,
        rebuild_source_path=args.rebuild_source_path,
        rebuild_source_collection_name=args.rebuild_source_collection_name,
        rebuild_insert_batch_size=args.rebuild_insert_batch_size,
        rebuild_embedding_batch_size=args.rebuild_embedding_batch_size,
    )
    store.initialize(mode=args.mode, clean_before_init=args.clean_before_init)
    return store


def run_server(args: argparse.Namespace) -> None:
    store = _build_store_from_args(args)
    context = SimpleNamespace(store=store, lock=threading.RLock(), task_name=args.task_name)
    server = ThreadingHTTPServer((args.host, args.port), _MemoryServiceHandler)
    server.context = context
    try:
        server.serve_forever()
    finally:
        server.server_close()
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--store-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="init_if_missing")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retrieve-key", type=str, default="memory_text")
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--rebuild-source-path", type=str, default=None)
    parser.add_argument("--rebuild-source-collection-name", type=str, default=None)
    parser.add_argument("--embedding-api-url", type=str, default=None)
    parser.add_argument("--embedding-api-key", type=str, default="empty")
    parser.add_argument("--embedding-model", type=str, default="bge_m3")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--rebuild-insert-batch-size", type=int, default=1000)
    parser.add_argument("--rebuild-embedding-batch-size", type=int, default=256)
    parser.add_argument("--only-successful", action="store_true")
    parser.add_argument("--clean-before-init", action="store_true")
    args = parser.parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
