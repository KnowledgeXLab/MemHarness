# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus

import requests

from .milvus_store import MilvusMemoryStore


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
    def __init__(
        self,
        process: subprocess.Popen | None,
        base_url: str,
        log_path: str,
        slurm_job_id: str | None = None,
    ) -> None:
        self.process = process
        self.base_url = base_url
        self.log_path = log_path
        self.slurm_job_id = slurm_job_id
        self._closed = False
        atexit.register(self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.slurm_job_id:
            subprocess.run(
                ["scancel", str(self.slurm_job_id)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.slurm_job_id = None
            return
        if self.process is None:
            return
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def _spawn_config_dict(
    *,
    bind_host: str,
    port: int,
    task_name: str,
    memory_dir: str,
    memory_config,
) -> dict:
    """JSON-serializable config for ``--config-json`` (matches argparse destinations)."""
    d: dict = {
        "host": bind_host,
        "port": port,
        "task_name": task_name,
        "store_dir": memory_dir,
        "mode": str(memory_config.mode),
        "timeout": int(memory_config.vdb_timeout),
        "retrieve_key": str(memory_config.retrieve_key),
        "embedding_dim": int(memory_config.embedding_dim),
        "top_k": int(memory_config.top_k),
        "min_score": float(memory_config.min_retrieval_score),
        "embedding_api_key": str(memory_config.embedding_api_key),
        "embedding_model": str(memory_config.embedding_model),
        "only_successful": bool(memory_config.get("only_successful", True)),
        "clean_before_init": bool(memory_config.get("clean_before_init", False)),
        "rebuild_insert_batch_size": int(memory_config.get("rebuild_insert_batch_size") or 1000),
        "rebuild_embedding_batch_size": int(memory_config.get("rebuild_embedding_batch_size") or 256),
    }
    for key, cfg_name in [
        ("collection_name", "collection_name"),
        ("rebuild_source_path", "rebuild_source_path"),
        ("rebuild_source_collection_name", "rebuild_source_collection_name"),
        ("embedding_api_url", "embedding_api_url"),
    ]:
        val = getattr(memory_config, cfg_name, None)
        if val:
            d[key] = str(val)
    return d


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
    rl = memory_config.get("remote_slurm_launch") if hasattr(memory_config, "get") else None
    remote_on = rl is not None and bool(rl.get("enable", False))
    # 远程 Slurm：端口写入 spawn JSON 并在 DB 节点 bind；勿在提交机上 _find_free_port。
    if remote_on:
        ro_port = rl.get("server_port") if hasattr(rl, "get") else None
        if ro_port is not None and int(ro_port) > 0:
            port = int(ro_port)
        elif requested_port > 0:
            port = requested_port
        else:
            port = 8765
    else:
        port = requested_port if requested_port > 0 else _find_free_port(bind_host)
    startup_timeout = max(1, int(memory_config.server_startup_timeout or 60))
    log_path = os.path.join(memory_dir, "memory_server.log")

    spawn_cfg = _spawn_config_dict(
        bind_host=bind_host,
        port=port,
        task_name=task_name,
        memory_dir=memory_dir,
        memory_config=memory_config,
    )

    if remote_on:
        from .remote_slurm_launcher import start_memory_vdb_on_slurm_node

        return start_memory_vdb_on_slurm_node(
            memory_config=memory_config,
            memory_dir=memory_dir,
            spawn_config=spawn_cfg,
            startup_timeout=startup_timeout,
            log_path=log_path,
        )

    config_json_path = os.path.join(memory_dir, "memory_server_spawn_config.json")
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(spawn_cfg, f, indent=2)

    command = [
        sys.executable,
        "-m",
        "agent_system.memory.local_service",
        "--config-json",
        config_json_path,
    ]

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


def _build_store_from_args(args: argparse.Namespace) -> MilvusMemoryStore:
    store = MilvusMemoryStore(
        task_name=args.task_name,
        store_dir=args.store_dir,
        collection_name=getattr(args, "collection_name", None),
        embedding_api_url=getattr(args, "embedding_api_url", None),
        embedding_api_key=getattr(args, "embedding_api_key", "empty"),
        embedding_model=getattr(args, "embedding_model", "bge_m3"),
        embedding_dim=args.embedding_dim,
        timeout=args.timeout,
        only_successful=bool(getattr(args, "only_successful", False)),
        top_k=args.top_k,
        min_score=args.min_score,
        retrieve_key=args.retrieve_key,
        rebuild_source_path=getattr(args, "rebuild_source_path", None),
        rebuild_source_collection_name=getattr(args, "rebuild_source_collection_name", None),
        rebuild_insert_batch_size=int(getattr(args, "rebuild_insert_batch_size", 1000)),
        rebuild_embedding_batch_size=int(getattr(args, "rebuild_embedding_batch_size", 256)),
    )
    store.initialize(mode=args.mode, clean_before_init=bool(getattr(args, "clean_before_init", False)))
    return store


def _namespace_from_config_json(path: str) -> argparse.Namespace:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config-json must contain a JSON object")
    return argparse.Namespace(**data)


def run_server(args: argparse.Namespace) -> None:
    import uvicorn

    from .memory_fastapi import create_memory_fastapi_app

    store = _build_store_from_args(args)
    lock = threading.RLock()
    app = create_memory_fastapi_app(store, args.task_name, lock)
    try:
        uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    finally:
        store.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", type=str, default=None, help="Load all server options from JSON (overrides other flags).")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--store-dir", type=str, default=None)
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
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.config_json:
        args = _namespace_from_config_json(args.config_json)
    else:
        if args.port is None or args.task_name is None or args.store_dir is None:
            parser.error("--port, --task-name and --store-dir are required without --config-json")
    run_server(args)


if __name__ == "__main__":
    main()
