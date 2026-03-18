from __future__ import annotations

import os
import uuid
from typing import Iterable

import torch.distributed as dist
from omegaconf import DictConfig

from verl.memory.memory_store import VectorMemoryStore
from verl.memory.types import MemoryEvent, MemoryRecord, RetrievedMemory


def truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_text(text: str, max_chars: int) -> str:
    collapsed = " ".join((text or "").split())
    return truncate_text(collapsed, max_chars=max_chars)


class RolloutMemoryManager:
    def __init__(
        self,
        memory_config: DictConfig | None,
        task_name: str,
        rollout_log_dir: str | None,
        rank: int,
    ) -> None:
        self.config = memory_config
        self.task_name = task_name
        self.rollout_log_dir = rollout_log_dir
        self.rank = rank
        self.current_step: int | None = None
        self.enabled = bool(memory_config and memory_config.enabled)
        self.store = None
        self.writer_rank = int(memory_config.writer_rank) if self.enabled else 0
        self.can_write = self.enabled and self.rank == self.writer_rank


        if not self.enabled:
            return

        memory_dir = memory_config.store_dir
        if not memory_dir:
            base_dir = rollout_log_dir or os.getcwd()
            memory_dir = os.path.join(base_dir, "memory_store")
        os.makedirs(memory_dir, exist_ok=True)

        self.mode = memory_config.mode
        self.store = VectorMemoryStore(
            backend=memory_config.backend,
            base_url=memory_config.vdb_base_url,
            timeout=memory_config.vdb_timeout,
            only_successful=memory_config.only_successful,
            top_k=memory_config.top_k,
            min_score=memory_config.min_retrieval_score,
            task_name=task_name,
            store_dir=memory_dir,
            collection_name=memory_config.collection_name,
            embedding_api_url=memory_config.embedding_api_url,
            embedding_api_key=memory_config.embedding_api_key,
            embedding_model=memory_config.embedding_model,
            embedding_dim=memory_config.embedding_dim,
            retrieve_key=memory_config.retrieve_key,
            can_write=self.can_write,
            rebuild_source_path=memory_config.rebuild_source_path,
            rebuild_source_collection_name=memory_config.rebuild_source_collection_name,
            rebuild_insert_batch_size=memory_config.rebuild_insert_batch_size,
            rebuild_embedding_batch_size=memory_config.rebuild_embedding_batch_size
        )
        self.store.initialize(mode=self.mode, clean_before_init=memory_config.clean_before_init)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def refresh(self, current_step: int | None = None) -> None:
        self.current_step = self._normalize_step(current_step)
        if self.store is not None:
            self.store.sync(current_step=self.current_step)

    def build_memory_message(self, state_text: str, round_idx: int, query_text: str | None = None) -> tuple[str, MemoryEvent | None]:
        if self.store is None:
            return "", None

        if query_text is None:
            query_text = normalize_text(state_text, max_chars=self.config.max_query_chars)
        if not query_text:
            return "", None

        retrieved = self.store.retrieve(query_text=query_text)
        if not retrieved:
            return "", None

        self._mark_retrieved_memories(retrieved)

        injected_text = self._format_memory_prompt(retrieved)
        event = MemoryEvent(
            event_type="retrieval",
            round_idx=round_idx,
            query_text=query_text,
            state_text=normalize_text(state_text, max_chars=self.config.max_state_chars),
            injected_text=injected_text,
            retrieved=[memory.to_dict() for memory in retrieved],
        )
        return injected_text, event

    # TODO: Implement this function with extractor (e.g. LLM)
    def build_records_from_episode(
        self,
        item_id: int,
        episode_id: str,
        messages: Iterable,
        ignored_user_contents: set[str],
        ignored_assistant_contents: set[str],
        score: float,
        global_step: int | str | None,
    ) -> list[MemoryRecord]:
        if self.store is None:
            return []
        if self.config.write_back is False:
            return []
        if self.config.only_store_positive_reward and score <= 0:
            return []

        max_pairs = self.config.max_pairs_per_episode
        state_max_chars = self.config.max_state_chars
        action_max_chars = self.config.max_action_chars
        memory_max_chars = self.config.max_memory_chars

        records: list[MemoryRecord] = []
        user_message = None
        source_step = 0
        message_list = list(messages)
        for message in message_list[2:]:
            if message.role == "user":
                if getattr(message, "message_type", "generic") != "env_state":
                    user_message = None
                    continue
                if message.content in ignored_user_contents:
                    user_message = None
                    continue
                user_message = message.content
                continue
            if message.role != "assistant" or user_message is None:
                continue
            if getattr(message, "message_type", "generic") != "env_action":
                user_message = None
                continue
            if message.content in ignored_assistant_contents:
                user_message = None
                continue
            source_step += 1
            state_text = normalize_text(user_message, max_chars=state_max_chars)
            action_text = normalize_text(message.content, max_chars=action_max_chars)
            if not state_text or not action_text:
                user_message = None
                continue
            memory_text = self._compose_memory_text(state_text=state_text, action_text=action_text)
            records.append(
                MemoryRecord(
                    memory_id=str(uuid.uuid4()),
                    task_name=self.task_name,
                    item_id=item_id,
                    source_episode_id=episode_id,
                    source_step=source_step,
                    state_text=state_text,
                    action_text=action_text,
                    memory_text=truncate_text(memory_text, max_chars=memory_max_chars),
                    reward=score,
                    success=score > 0,
                    created_step=global_step,
                    metadata={"rank": self.rank},
                )
            )
            user_message = None
            if len(records) >= max_pairs:
                break
        return records

    def add_records(self, records: list[MemoryRecord]) -> None:
        if self.store is None:
            return

        merged_records = self._gather_records_to_writer(records)
        if self.can_write and merged_records:
            self.store.add_records(merged_records)

    def finalize_step(self, current_step: int | None) -> None:
        self.current_step = self._normalize_step(current_step)
        if self.store is not None:
            self.store.sync(current_step=self.current_step)

    def update_memory_records(self, updates: list[dict]) -> int:
        if self.store is None:
            return 0
        return self.store.update_records(updates)

    # TODO
    def _compose_memory_text(self, state_text: str, action_text: str) -> str:
        return (
            "Past successful experience:\n"
            f"- Similar state: {state_text}\n"
            f"- Helpful action or principle: {action_text}"
        )

    # TODO: support more parameter combination of memory prompt
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

    @staticmethod
    def _normalize_step(current_step: int | str | None) -> int | None:
        if current_step is None:
            return None
        if isinstance(current_step, int):
            return current_step
        try:
            return int(current_step)
        except (TypeError, ValueError):
            return None

    def _gather_records_to_writer(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        if not records and not (dist.is_available() and dist.is_initialized()):
            return []
        if not (dist.is_available() and dist.is_initialized()):
            return records

        world_size = dist.get_world_size()
        gathered_records: list[list[MemoryRecord] | None] = [None] * world_size
        dist.all_gather_object(gathered_records, records)

        if not self.can_write:
            return []

        merged_records: list[MemoryRecord] = []
        for rank_records in gathered_records:
            if rank_records:
                merged_records.extend(rank_records)
        return merged_records

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    def _mark_retrieved_memories(self, retrieved: list[RetrievedMemory]) -> None:
        # Skip updating retrieval stats in read-only mode (testing/evaluation)
        if not self.can_write or self.config.write_back is False:
            return
        updates: list[dict] = []
        last_used_step = self.current_step
        for memory in retrieved:
            updates.append(
                {
                    "memory_id": memory.memory_id,
                    "_increment_retrieval_count": 1,
                    "last_used_step": last_used_step,
                }
            )
        if updates:
            self.update_memory_records(updates)
