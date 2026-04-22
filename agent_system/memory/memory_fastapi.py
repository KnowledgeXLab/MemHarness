# Copyright 2024 MemAdaptor memory HTTP service (FastAPI).
"""FastAPI application exposing the same HTTP API as the legacy stdlib server."""

from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .milvus_store import MilvusMemoryStore
from .types import MemoryRecord


def create_memory_fastapi_app(
    store: MilvusMemoryStore,
    task_name: str,
    lock: threading.RLock,
) -> FastAPI:
    app = FastAPI(title="MemAdaptor Memory VDB", version="1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "task_name": task_name,
            "collection_name": store.collection_name,
            "store_dir": store.store_dir,
        }

    @app.get("/memories/count")
    async def memories_count() -> dict[str, Any]:
        try:
            with lock:
                n = store.count_records()
            return {"count": int(n)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/memories/batch")
    async def memories_batch(body: dict) -> dict[str, Any]:
        items = body.get("items", [])
        if not isinstance(items, list):
            raise HTTPException(status_code=400, detail="items must be a list")
        try:
            records = [MemoryRecord.from_dict(item) for item in items if isinstance(item, dict)]
            with lock:
                store.add_records(records)
            return {"inserted": len(records)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/memories/search")
    async def memories_search(body: dict) -> JSONResponse:
        query_text = str(body.get("query", "") or "")
        if not query_text:
            return JSONResponse(content=[])
        try:
            with lock:
                old_top_k = store.top_k
                old_min_score = store.min_score
                old_only_successful = store.only_successful
                store.top_k = int(body.get("top_k", old_top_k))
                store.min_score = float(body.get("min_score", old_min_score))
                store.only_successful = bool(body.get("only_successful", old_only_successful))
                try:
                    retrieved = store.retrieve(query_text=query_text)
                finally:
                    store.top_k = old_top_k
                    store.min_score = old_min_score
                    store.only_successful = old_only_successful
            return JSONResponse(content=[item.to_dict() for item in retrieved])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/memories/update")
    async def memories_update(body: dict) -> dict[str, Any]:
        updates = body.get("updates", [])
        if not isinstance(updates, list):
            raise HTTPException(status_code=400, detail="updates must be a list")
        try:
            with lock:
                updated = store.update_records(updates)
            return {"updated": updated}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/memories/prune")
    async def memories_prune(body: dict) -> dict[str, Any]:
        try:
            score_threshold = float(body.get("score_threshold", 0.0))
            min_uses = int(body.get("min_uses", 0))
            with lock:
                deleted = store.prune_low_utility_memories(score_threshold, min_uses)
            return {"deleted": deleted}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/memories/clear")
    async def memories_clear() -> dict[str, Any]:
        try:
            with lock:
                store.clear_all_records()
            return {"status": "ok"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
