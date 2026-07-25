from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

from .experience_utility import (
    UTILITY_SUCC_KEY,
    UTILITY_USE_KEY,
    collect_memory_ids_from_info_list,
    episode_success_from_batch,
)
from .global_step_schedule import match_phase_for_global_step
from .memory_store import MemoryStoreDispatcher
from .types import MEMORY_STATE_UNAVAILABLE_PLACEHOLDER, MemoryEvent, RetrievedMemory


# Aligns with ``env.memory.empty_retrieval_message`` default in ``verl/trainer/config/ppo_trainer.yaml``.
_DEFAULT_EMPTY_RETRIEVAL_MESSAGE = (
    "(No validated memory principle applies; rely on observation and reasoning.)"
)


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
        self._trainer_global_step: int | None = None
        self.enabled = bool(memory_config.enabled)
        self.store = None
        self._timing_retrieve_sec: float = 0.0
        self._timing_retrieve_calls: int = 0
        self._timing_retrieve_hits_sum: int = 0
        self._timing_retrieval_utility_sec: float = 0.0
        self._timing_retrieval_utility_calls: int = 0
        self._rollup_retrieve_sec: float = 0.0
        self._rollup_retrieve_calls: int = 0
        self._rollup_retrieve_hits_sum: int = 0
        self._rollup_records_inserted: int = 0
        self._rollup_batch_insert_sec: float = 0.0
        self._rollup_prune_deleted: int = 0
        self._rollup_prune_sec: float = 0.0

        if not self.enabled:
            return

        backend = str(memory_config.backend).strip().lower()
        if backend != "milvus" and backend != "http":
            raise ValueError(f"Unsupported memory backend for verl-agent: {backend}")

        memory_dir = memory_config.store_dir
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
            memory_text_retrieval_dedupe_similarity_threshold=memory_config.get(
                "memory_text_retrieval_dedupe_similarity_threshold"
            ),
            memory_text_retrieval_dedupe_prefetch_limit=int(
                memory_config.get("memory_text_retrieval_dedupe_prefetch_limit") or 24
            ),
            memory_text_insert_dedupe_similarity_threshold=memory_config.get(
                "memory_text_insert_dedupe_similarity_threshold"
            ),
            memory_text_insert_dedupe_probe_limit=int(
                memory_config.get("memory_text_insert_dedupe_probe_limit") or 40
            ),
        )
        self.store.initialize(
            mode=memory_config.mode,
            clean_before_init=memory_config.clean_before_init,
        )
        print(f"Memory VDB row count: {self.store.count_records()}", flush=True)

        self._last_utility_prune_step: int = -1
        self._last_utility_clear_step: int = -1

        retrieval_mode = str(memory_config.retrieval_mode).strip().lower()
        retrieve_key = str(memory_config.retrieve_key).strip().lower()
        if retrieval_mode == "fixed" and retrieve_key in ("memory_text", "memory"):
            print(
                "Memory VDB: retrieval_mode=fixed uses current observation text as the query embedding, "
                "but retrieve_key=%r means vectors are built from that field (not state_text). "
                "If you want to use the same field for both query and index, set retrieve_key to state_text and rebuild the VDB.",
                retrieve_key,
                flush=True,
            )

    def refresh(self) -> None:
        return

    def set_trainer_global_step(self, step: int | None) -> None:
        """Set current trainer step for ``retrieval_mode_phases`` (rollout sets this once per batch)."""
        self._trainer_global_step = int(step) if step is not None else None

    def retrieval_mode(self) -> str:
        if not self.enabled:
            return "disabled"
        phase = match_phase_for_global_step(
            self.config.get("retrieval_mode_phases"),
            self._trainer_global_step,
        )
        if phase is not None and phase.get("mode") is not None:
            return str(phase["mode"]).strip().lower()
        return str(self.config.retrieval_mode).strip().lower()

    def is_fixed_mode(self) -> bool:
        return self.retrieval_mode() == "fixed"

    def is_agentic_mode(self) -> bool:
        return self.retrieval_mode() == "agentic"

    def _experience_utility_cfg(self) -> dict:
        if not self.enabled or self.config is None:
            return {}
        eu = self.config.get("experience_utility")
        if eu is None:
            return {}
        return OmegaConf.to_container(eu, resolve=True) or {}

    def experience_utility_enabled(self) -> bool:
        return bool(self._experience_utility_cfg().get("enable"))

    def _log_operation_timing(self) -> bool:
        if self.config is None:
            return False
        return bool(self.config.get("log_operation_timing", True))

    def _empty_retrieval_injected_text(self) -> str:
        if self.config is None:
            return _DEFAULT_EMPTY_RETRIEVAL_MESSAGE
        raw = self.config.get("empty_retrieval_message")
        msg = str(raw).strip() if raw is not None else ""
        return msg if msg else _DEFAULT_EMPTY_RETRIEVAL_MESSAGE

    def _merge_retrieve_window_into_rollup(self) -> None:
        if not self.enabled:
            return
        if self._timing_retrieve_calls <= 0:
            self._timing_retrieve_sec = 0.0
            self._timing_retrieve_calls = 0
            self._timing_retrieve_hits_sum = 0
            return
        self._rollup_retrieve_sec += self._timing_retrieve_sec
        self._rollup_retrieve_calls += self._timing_retrieve_calls
        self._rollup_retrieve_hits_sum += self._timing_retrieve_hits_sum
        self._timing_retrieve_sec = 0.0
        self._timing_retrieve_calls = 0
        self._timing_retrieve_hits_sum = 0

    def _flush_retrieval_rollup_timings(self, trainer_global_step: int | None) -> None:
        """Log retrieve / retrieval-utility latency for this rollout window, then merge retrieve stats for W&B."""
        if not self.enabled:
            return
        step_s = "" if trainer_global_step is None else str(int(trainer_global_step))
        if self._log_operation_timing():
            if self._timing_retrieve_calls:
                mean = self._timing_retrieve_sec / max(1, self._timing_retrieve_calls)
                print(
                    f"memory.timing op=retrieve_aggregate trainer_global_step={step_s} "
                    f"calls={self._timing_retrieve_calls} total_sec={self._timing_retrieve_sec:.4f} "
                    f"mean_sec={mean:.4f}",
                    flush=True,
                )
            if self._timing_retrieval_utility_calls:
                mean_u = self._timing_retrieval_utility_sec / max(1, self._timing_retrieval_utility_calls)
                print(
                    f"memory.timing op=retrieval_utility_update_aggregate trainer_global_step={step_s} "
                    f"calls={self._timing_retrieval_utility_calls} "
                    f"total_sec={self._timing_retrieval_utility_sec:.4f} mean_sec={mean_u:.4f}",
                    flush=True,
                )
        self._merge_retrieve_window_into_rollup()
        self._timing_retrieval_utility_sec = 0.0
        self._timing_retrieval_utility_calls = 0

    def pop_rollout_memory_metrics(self) -> dict[str, float]:
        """Finalize metrics for the last trainer step (including any retrieve window not yet flushed)."""
        if not self.enabled:
            return {}
        self._merge_retrieve_window_into_rollup()
        rc = self._rollup_retrieve_calls
        rs = self._rollup_retrieve_sec
        rh = self._rollup_retrieve_hits_sum
        out: dict[str, float] = {}
        if rc > 0:
            out["memory/retrieve_calls"] = float(rc)
            out["memory/retrieve_total_sec"] = float(rs)
            out["memory/retrieve_mean_sec"] = float(rs / rc)
            out["memory/retrieve_avg_hits"] = float(rh / rc)
        else:
            out["memory/retrieve_calls"] = 0.0
            out["memory/retrieve_total_sec"] = 0.0
            out["memory/retrieve_mean_sec"] = 0.0
            out["memory/retrieve_avg_hits"] = 0.0
        out["memory/records_inserted"] = float(self._rollup_records_inserted)
        out["memory/batch_insert_sec"] = float(self._rollup_batch_insert_sec)
        out["memory/prune_deleted"] = float(self._rollup_prune_deleted)
        out["memory/prune_sec"] = float(self._rollup_prune_sec)
        nrow = -1.0
        if self.store is not None:
            try:
                c = int(self.store.count_records())
                if c >= 0:
                    nrow = float(c)
            except Exception:
                pass
        out["memory/vdb_row_count"] = nrow
        self._rollup_retrieve_sec = 0.0
        self._rollup_retrieve_calls = 0
        self._rollup_retrieve_hits_sum = 0
        self._rollup_records_inserted = 0
        self._rollup_batch_insert_sec = 0.0
        self._rollup_prune_deleted = 0
        self._rollup_prune_sec = 0.0
        return out

    def _apply_retrieval_utility_updates(self, retrieved: list[RetrievedMemory]) -> None:
        if not self.experience_utility_enabled() or self.store is None:
            return
        seen: set[str] = set()
        updates: list[dict] = []
        for memory in retrieved:
            mid = (memory.memory_id or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            updates.append(
                {
                    "memory_id": mid,
                    "_metadata_int_increment": {UTILITY_USE_KEY: 1},
                    "_sync_value_from_utility": True,
                }
            )
        if updates:
            t0 = time.perf_counter()
            self.store.update_records(updates)
            dt = time.perf_counter() - t0
            if self._log_operation_timing():
                self._timing_retrieval_utility_sec += dt
                self._timing_retrieval_utility_calls += 1

    def _apply_episode_utility_updates(
        self,
        total_infos: list,
        success: dict,
        trainer_global_step: int | None,
    ) -> None:
        if not self.experience_utility_enabled() or self.store is None:
            return
        if not total_infos or not success:
            return
        vu_step = "" if trainer_global_step is None else str(int(trainer_global_step))
        batch_size = len(total_infos)
        succ_deltas: dict[str, int] = defaultdict(int)
        for env_i in range(batch_size):
            if not episode_success_from_batch(success, env_i):
                continue
            infos_for_env = total_infos[env_i]
            if not isinstance(infos_for_env, list):
                continue
            for mid in collect_memory_ids_from_info_list(infos_for_env):
                succ_deltas[mid] += 1
        updates = [
            {
                "memory_id": mid,
                "_metadata_int_increment": {UTILITY_SUCC_KEY: int(delta)},
                "_sync_value_from_utility": True,
                "value_update_step": vu_step,
            }
            for mid, delta in succ_deltas.items()
            if mid and delta > 0
        ]
        if updates:
            t0 = time.perf_counter()
            self.store.update_records(updates)
            if self._log_operation_timing():
                print(
                    f"memory.timing op=episode_utility_update trainer_global_step={vu_step} "
                    f"rows={len(updates)} elapsed_sec={time.perf_counter() - t0:.4f}",
                    flush=True,
                )

    def _maybe_run_experience_utility_maintenance(self, trainer_global_step: int | None) -> None:
        if not self.experience_utility_enabled() or self.store is None:
            return
        if trainer_global_step is None:
            return
        step = int(trainer_global_step)
        if step <= 0:
            return
        cfg = self._experience_utility_cfg()
        clear_every = int(cfg.get("clear_every_n_global_steps") or 0)
        if clear_every > 0 and step % clear_every == 0 and step != self._last_utility_clear_step:
            t0 = time.perf_counter()
            try:
                self.store.clear_all_records()
            except Exception as exc:
                self._last_utility_clear_step = step
                self._last_utility_prune_step = step
                print(
                    f"experience_utility: clear FAILED at global_step={step} (training continues): {exc}",
                    flush=True,
                )
                return
            dt = time.perf_counter() - t0
            self._last_utility_clear_step = step
            self._last_utility_prune_step = step
            print(f"experience_utility: cleared memory store at global_step={step}", flush=True)
            if self._log_operation_timing():
                print(
                    f"memory.timing op=utility_clear trainer_global_step={step} elapsed_sec={dt:.4f}",
                    flush=True,
                )
            return

        prune_every = int(cfg.get("prune_every_n_global_steps") or 0)
        if prune_every > 0 and step % prune_every == 0 and step != self._last_utility_prune_step:
            threshold = float(cfg.get("prune_score_threshold", 0.35))
            min_uses = int(cfg.get("min_uses_before_prune", 3))
            t0 = time.perf_counter()
            try:
                deleted = self.store.prune_low_utility_memories(threshold, min_uses)
            except Exception as exc:
                self._last_utility_prune_step = step
                print(
                    f"experience_utility: prune FAILED at global_step={step} (training continues): {exc}",
                    flush=True,
                )
                return
            dt = time.perf_counter() - t0
            self._rollup_prune_deleted += int(deleted)
            self._rollup_prune_sec += float(dt)
            self._last_utility_prune_step = step
            print(
                f"experience_utility: pruned {deleted} memories (threshold={threshold} min_uses={min_uses}) "
                f"at global_step={step}",
                flush=True,
            )
            if self._log_operation_timing():
                print(
                    f"memory.timing op=utility_prune trainer_global_step={step} deleted={deleted} "
                    f"elapsed_sec={dt:.4f}",
                    flush=True,
                )

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
        empty_msg = self._empty_retrieval_injected_text()
        if self.store is None:
            return "", None

        if query_text is None:
            query_text = normalize_text(
                state_text,
                max_chars=self.config.max_query_chars,
            )
        if not query_text:
            return "", None

        if bool(self.config.get("debug_retrieval", False)):
            print(
                f"memory.retrieve preview: trainer_global_step={self._trainer_global_step} "
                f"resolved_retrieval_mode={self.retrieval_mode()} query_len={len(query_text)} "
                f"query_preview={truncate_text(query_text, 120)!r}",
                flush=True,
            )

        t0 = time.perf_counter()
        retrieved = self.store.retrieve(query_text=query_text)
        self._timing_retrieve_sec += time.perf_counter() - t0
        self._timing_retrieve_calls += 1
        self._timing_retrieve_hits_sum += len(retrieved)

        norm_state = normalize_text(state_text, max_chars=self.config.max_state_chars)
        if not retrieved:
            event = MemoryEvent(
                event_type="retrieval",
                query_text=query_text,
                state_text=norm_state,
                injected_text=empty_msg,
                retrieved=[],
            )
            return empty_msg, event

        self._apply_retrieval_utility_updates(retrieved)

        injected_text = self._format_memory_prompt(retrieved)
        event = MemoryEvent(
            event_type="retrieval",
            query_text=query_text,
            state_text=norm_state,
            injected_text=injected_text,
            retrieved=[memory.to_dict() for memory in retrieved],
        )
        return injected_text, event

    def build_memory_event(self, state_text: str, query_text: str | None = None) -> MemoryEvent | None:
        _, event = self.build_memory_message_and_event(state_text=state_text, query_text=query_text)
        return event

    def maybe_write_rollout_memories(self, **kwargs) -> list:
        """
        After parallel rollouts: optionally summarize trajectories and insert into the memory store.

        Expected kwargs from ``TrajectoryCollector`` / env manager:
        ``config``, ``tokenizer``, ``actor_rollout_wg`` (for mode=self),
        ``total_batch_list``, ``total_infos``, ``episode_rewards``, ``episode_lengths``,
        ``success``, ``traj_uid``, optional ``grpo_group_uid`` (same layout as rollout env batch).

        Heavy lifting lives in ``agent_system.memory.experience_summarizer`` to keep this file readable.
        """
        from agent_system.memory.experience_summarizer import maybe_summarize_and_write_experiences

        if not self.enabled:
            return []

        config = kwargs.get("config")
        tokenizer = kwargs.get("tokenizer")
        total_batch_list = kwargs.get("total_batch_list")
        episode_rewards = kwargs.get("episode_rewards")
        episode_lengths = kwargs.get("episode_lengths")
        traj_uid = kwargs.get("traj_uid")
        if config is None or tokenizer is None:
            return []
        if not total_batch_list or episode_rewards is None or episode_lengths is None or traj_uid is None:
            return []

        total_infos = kwargs.get("total_infos") or []
        success = kwargs.get("success") or {}
        trainer_global_step = kwargs.get("trainer_global_step")

        self._flush_retrieval_rollup_timings(trainer_global_step)

        n_ins, ins_sec = maybe_summarize_and_write_experiences(
            config=config,
            memory_manager=self,
            tokenizer=tokenizer,
            actor_rollout_wg=kwargs.get("actor_rollout_wg"),
            total_batch_list=total_batch_list,
            total_infos=total_infos,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            trainer_global_step=trainer_global_step,
            grpo_group_uid=kwargs.get("grpo_group_uid"),
        )
        self._rollup_records_inserted += int(n_ins)
        self._rollup_batch_insert_sec += float(ins_sec)

        self._apply_episode_utility_updates(
            total_infos=total_infos,
            success=success,
            trainer_global_step=trainer_global_step,
        )
        self._maybe_run_experience_utility_maintenance(trainer_global_step)
        return []

    def sample_random_state_texts(
        self,
        n: int,
        exclude_memory_ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Sample decoy ``state_text`` rows from the memory store (for adaptor ablations)."""
        need = max(0, int(n))
        if need == 0 or not self.enabled or self.store is None:
            return [""] * need
        try:
            return self.store.sample_random_state_texts(need, exclude_memory_ids=exclude_memory_ids)
        except Exception as exc:
            print(f"memory.sample_random_state_texts failed (returning empty): {exc}", flush=True)
            return [""] * need

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    def _format_memory_prompt(
        self,
        retrieved: list[RetrievedMemory],
        append_score=False,
        append_action=False,
        append_value=False,
        append_state=True,
    ) -> str:
        """Format retrieved memories for injection into the observation.

        Each hit is one block: ``--- Memory k ---`` then labeled lines (Situation / Principle / …).
        ``scripts/mem_fail_judge.parse_injected_memory_lines`` parses ``Principle:`` lines per block.
        """
        raw_header = self.config.get("prompt_header")
        header = str(raw_header).strip() if raw_header is not None else ""
        lines: list[str] = []
        if header:
            lines.append(header)
        for index, memory in enumerate(retrieved, start=1):
            if lines:
                lines.append("")
            lines.append(f"--- Memory {index} ---")
            if append_state:
                st = (memory.state_text or "").strip()
                if not st:
                    st = MEMORY_STATE_UNAVAILABLE_PLACEHOLDER
                lines.append(f"Situation: {st}")
            lines.append(f"Principle: {memory.memory_text}")
            if append_action and (memory.action_text or "").strip():
                lines.append(f"Recorded action: {memory.action_text}")
            if append_score:
                lines.append(f"Retrieval score: {memory.score:.3f}")
            if append_value and memory.value is not None:
                lines.append(f"Memory value: {float(memory.value):.3f}")
        return "\n".join(lines)
