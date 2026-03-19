from __future__ import annotations

import os
import re

from omegaconf import DictConfig

from .memory_store import MemoryStoreDispatcher
from .types import MemoryEvent, RetrievedMemory


def truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_text(text: str, max_chars: int) -> str:
    collapsed = " ".join((text or "").split())
    return truncate_text(collapsed, max_chars=max_chars)


class MemoryManager:
    """Config-driven external memory manager used by environment managers."""

    def __init__(self, memory_config: DictConfig | None, task_name: str) -> None:
        self.config = memory_config
        self.task_name = task_name
        self.enabled = bool(memory_config and memory_config.get("enabled", False))
        self.store = None

        if not self.enabled:
            return

        backend = str(memory_config.backend).strip().lower()
        if backend != "milvus" and backend != "http":
            raise ValueError(f"Unsupported memory backend for verl-agent: {backend}")

        memory_dir = memory_config.get("store_dir")
        if not memory_dir:
            memory_dir = os.path.join(os.getcwd(), "memory_store", task_name.replace("/", "_"))

        self.store = MemoryStoreDispatcher(
            backend=memory_config.backend,
            base_url=memory_config.vdb_base_url,
            task_name=task_name,
            store_dir=memory_dir,
            collection_name=memory_config.collection_name,
            embedding_api_url=memory_config.embedding_api_url,
            embedding_api_key=memory_config.embedding_api_key,
            embedding_model=memory_config.embedding_model,
            embedding_dim=memory_config.embedding_dim,
            timeout=memory_config.vdb_timeout,
            only_successful=memory_config.only_successful,
            top_k=memory_config.top_k,
            min_score=memory_config.min_retrieval_score,
            retrieve_key=memory_config.retrieve_key,
            rebuild_source_path=memory_config.rebuild_source_path,
            rebuild_source_collection_name=memory_config.rebuild_source_collection_name,
            rebuild_insert_batch_size=memory_config.rebuild_insert_batch_size,
            rebuild_embedding_batch_size=memory_config.rebuild_embedding_batch_size,
        )
        self.store.initialize(
            mode=memory_config.mode,
            clean_before_init=memory_config.clean_before_init,
        )

    def refresh(self) -> None:
        return

    def retrieval_mode(self) -> str:
        if not self.enabled:
            return "disabled"
        return str(self.config.get("retrieval_mode", "fixed")).strip().lower()

    def is_fixed_mode(self) -> bool:
        return self.retrieval_mode() == "fixed"

    def is_agentic_mode(self) -> bool:
        return self.retrieval_mode() == "agentic"

    def append_retrieval_hint(self, prompt: str) -> str:
        if not self.enabled or not self.is_agentic_mode():
            return prompt
        hint = self.config.retrieval_instruction_prompt.format(open_tag=self.config.retrieval_query_open_tag, close_tag=self.config.retrieval_query_close_tag)
        return f"{prompt}\n\n{hint}"

    def extract_query(self, response_text: str) -> str | None:
        if not self.enabled or not self.is_agentic_mode():
            return None
        open_tag = re.escape(self.config.retrieval_query_open_tag)
        close_tag = re.escape(self.config.retrieval_query_close_tag)
        match = re.search(f"{open_tag}(.*?){close_tag}", response_text or "", flags=re.DOTALL)
        if not match:
            return None
        query_text = normalize_text(match.group(1), max_chars=self.config.max_query_chars)
        return query_text or None

    def build_memory_message(self, state_text: str, query_text: str | None = None) -> str:
        injected_text, _ = self.build_memory_message_and_event(state_text=state_text, query_text=query_text)
        return injected_text

    def build_memory_message_and_event(self, state_text: str, query_text: str | None = None) -> tuple[str, MemoryEvent | None]:
        if self.store is None:
            return "", None

        if query_text is None:
            query_text = normalize_text(
                state_text,
                max_chars=self.config.max_query_chars,
            )
        if not query_text:
            return "", None

        retrieved = self.store.retrieve(query_text=query_text)
        if not retrieved:
            return "", None

        injected_text = self._format_memory_prompt(retrieved)
        event = MemoryEvent(
            event_type="retrieval",
            query_text=query_text,
            state_text=normalize_text(state_text, max_chars=self.config.max_state_chars),
            injected_text=injected_text,
            retrieved=[memory.to_dict() for memory in retrieved],
        )
        return injected_text, event

    def build_memory_event(self, state_text: str, query_text: str | None = None) -> MemoryEvent | None:
        _, event = self.build_memory_message_and_event(state_text=state_text, query_text=query_text)
        return event

    def maybe_write_rollout_memories(self, **kwargs) -> list:
        # Reserved hook for future online write-back. Pre-experiment does not use it yet.
        del kwargs
        return []

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    def _format_memory_prompt(self, retrieved: list[RetrievedMemory]) -> str:
        header = self.config.prompt_header
        lines = [header]
        for index, memory in enumerate(retrieved, start=1):
            lines.append(f"{index}. score={memory.score:.3f}")
            lines.append(f"   similar_state: {memory.state_text}")
            lines.append(f"   useful_action: {memory.action_text}")
            if memory.value is not None:
                lines.append(f"   memory_value: {memory.value:.3f}")
        return "\n".join(lines)
