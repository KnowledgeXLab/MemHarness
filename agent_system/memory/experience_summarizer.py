# Copyright 2025 MemAdaptor
#
# Online experience / memory write-back: LLM extracts JSON ``{"memories":[...]}`` per trajectory, then VDB insert.
# - mode=self: batch prompts to ``actor_rollout_wg.generate_sequences`` (same Reasoning rollout worker).
# - mode=teacher: OpenAI-compatible ``/v1/chat/completions`` on the driver (no Ray worker).

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import urllib.error
import urllib.request
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer

from agent_system.memory.memory_manager import MemoryManager, normalize_text, truncate_text
from agent_system.memory.types import MemoryRecord
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)

# Defaults aligned with ``scripts/extract_memory_records.py`` (JSON extraction schema).
DEFAULT_JSON_SYSTEM_PROMPT = """You are an expert agent-memory extraction system.

Your job is to read one successful trajectory from an interactive agent benchmark and extract a small set of reusable memories.

The extracted memories must satisfy all requirements:
1. Each memory should correspond to one local decision point or one short reusable principle grounded in the trajectory.
2. A memory must be useful for retrieval in a future similar state.
3. Do not summarize the whole trajectory into one global high-level paragraph.
4. Prefer step-level or subgoal-level memories.
5. Do not invent facts that are not supported by the trajectory.
6. The output must be valid JSON only, with no markdown fences and no extra explanation.
"""

DEFAULT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE = """Extract at most {num_memories} reusable memories from the following successful trajectory.

Task requirements:
- The benchmark is "{task_name}".
- The trajectory is successful, so the extracted memories should be treated as positive memories.
- Each memory should be suitable for naive retrieval later.
- Each memory should contain:
  - source_step: the step index in the trajectory where this memory is grounded. Use 1-based indexing over agent action turns.
  - state_text: a concise but sufficient description of the state/observation that would be useful as the retrieval key.
  - action_text: the actual helpful action from the trajectory, or the directly grounded action-oriented response.
  - memory_text: a short reusable memory written for future prompting. It should emphasize what to do in a similar local situation, and may mention important preconditions.
  - value: an optional initial quality/value estimate in [0, 1]. Use null if you are unsure.
  - metadata: an object that may include short fields such as "subgoal", "preconditions", "why_useful".

Important extraction rules:
- Keep memories local and concrete.
- Do not output duplicate memories.
- Do not output more than {num_memories} memories.
- If the trajectory is short, output fewer memories.
- Preserve benchmark-specific details when they are critical (e.g., object-state relations, environment preconditions, product constraints, or option-selection cues depending on the current task).

Return JSON with exactly this schema:
{{
  "memories": [
    {{
      "source_step": 1,
      "state_text": "...",
      "action_text": "...",
      "memory_text": "...",
      "metadata": {{
        "subgoal": "...",
        "preconditions": "...",
        "why_useful": "..."
      }}
    }}
  ]
}}

Trajectory metadata:
- dataset_item_id: {dataset_item_id}
- trajectory_index: {trajectory_index}
- episode_reward (online rollout): {episode_reward}
- episode_length (online rollout): {episode_length}
- success (online rollout): {success}
- traj_uid (online rollout): {traj_uid}

Trajectory:
{trajectory_text}
"""


def _exps_cfg(config: DictConfig) -> DictConfig:
    mc = OmegaConf.select(config, "env.memory.experience_summarizer")
    return mc if mc is not None else OmegaConf.create({})


def experience_summarizer_active(config: DictConfig) -> bool:
    mem = OmegaConf.select(config, "env.memory")
    if mem is None or not bool(mem.get("enabled", False)):
        return False
    if not bool(mem.get("write_back", False)):
        return False
    es = _exps_cfg(config)
    mode = str(es.get("mode", "none") or "none").strip().lower()
    return mode in ("self", "teacher")


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object parse; matches ``scripts/extract_memory_records.extract_json_object``."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _action_text_from_step(step: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> str:
    art = step.get("api_response_text")
    if isinstance(art, str) and art.strip():
        return art.strip()
    resp = step.get("responses")
    if isinstance(resp, torch.Tensor):
        return tokenizer.decode(resp.detach().cpu(), skip_special_tokens=True).strip()
    return ""


def _state_text_from_step(step: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> str:
    pr = step.get("prompts")
    if isinstance(pr, torch.Tensor):
        return tokenizer.decode(pr.detach().cpu(), skip_special_tokens=True).strip()
    iid = step.get("input_ids")
    if isinstance(iid, torch.Tensor):
        return tokenizer.decode(iid.detach().cpu(), skip_special_tokens=True).strip()
    return ""


def _format_trajectory_plain(
    steps: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    *,
    max_chars: int,
) -> str:
    lines: List[str] = []
    for t, step in enumerate(steps):
        if not step.get("active_masks", True):
            continue
        act = _action_text_from_step(step, tokenizer)
        st = _state_text_from_step(step, tokenizer)
        lines.append(f"--- Step {t + 1} ---")
        if st:
            lines.append(f"Context (prompt):\n{st}")
        if act:
            lines.append(f"Action (model output):\n{act}")
        rew = step.get("rewards")
        if rew is not None:
            lines.append(f"Step reward: {rew}")
    body = "\n\n".join(lines)
    return truncate_text(body, max_chars)


def _episode_success(success: Dict[str, np.ndarray], index: int) -> bool:
    if not success:
        return False
    for _k, arr in success.items():
        a = np.asarray(arr)
        if a.size <= index:
            continue
        try:
            return bool(np.asarray(a[index]).reshape(-1)[0] != 0)
        except Exception:
            return bool(a[index])
    return False


def _dataset_item_id_from_infos(total_infos: List[List[Dict[str, Any]]], env_index: int) -> Any:
    if env_index >= len(total_infos) or not total_infos[env_index]:
        return env_index
    last = total_infos[env_index][-1]
    if not isinstance(last, dict):
        return env_index
    for key in ("item_id", "dataset_item_id", "gamefile", "task_id"):
        if key in last and last[key] is not None:
            return last[key]
    return env_index


def _resolve_trajectory_user_prompt_template(es: DictConfig) -> str:
    t = es.get("trajectory_user_prompt_template")
    if t is not None and str(t).strip():
        return str(t)
    return ""


def _resolve_system_prompt(es: DictConfig) -> str:
    raw = es.get("system_prompt")
    s = str(raw).strip() if raw is not None else ""
    return s if s else DEFAULT_JSON_SYSTEM_PROMPT


def _build_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    user_content: str,
    apply_chat_template_kwargs: Optional[dict],
) -> str:
    apply_kw = dict(apply_chat_template_kwargs or {})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_kw,
    )


def _row_from_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_prompt_length: int,
    pad_token_id: int,
    truncation: str,
) -> dict:
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
        prompt=prompt,
        tokenizer=tokenizer,
        max_length=max_prompt_length,
        pad_token_id=pad_token_id,
        left_pad=True,
        truncation=truncation,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "position_ids": position_ids[0],
    }


def _normalize_chat_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base + "/chat/completions"


def _http_chat_completion(url: str, headers: Dict[str, str], payload: dict, timeout: float) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Chat API HTTP {e.code}: {err}") from e
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        return ""
    return str(content).strip()


def _summarize_batch_teacher(
    *,
    trajectory_user_messages: List[str],
    system_prompt: str,
    oa: DictConfig,
) -> List[str]:
    """Summarize trajectories using the teacher HTTP API."""

    base_url = str(oa["base_url"])
    url = _normalize_chat_url(base_url)
    api_key = str(oa["api_key"])
    if not api_key:
        raise ValueError(
            "env.memory.experience_summarizer.openai_api.api_key is empty; set it or OPENAI_API_KEY."
        )
    model = str(oa["model"])
    timeout = float(oa["timeout_sec"])
    max_tokens = int(oa["max_tokens"])
    temperature = float(oa["temperature"])
    max_concurrent = int(oa["max_concurrent"])
    max_retries = int(oa["max_retries"])
    retry_backoff = float(oa["retry_backoff_sec"])
    use_json_response = bool(oa.get("response_format_json", False))
    extra = oa["extra_headers"]
    extra_headers: Dict[str, str] = dict(OmegaConf.to_container(extra, resolve=True)) if extra else {}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        **extra_headers,
    }

    def one(i: int, user_text: str) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": max_tokens,
        }
        if temperature > 0:
            payload["temperature"] = temperature
        else:
            payload["temperature"] = 0
        if use_json_response:
            payload["response_format"] = {"type": "json_object"}

        delay = retry_backoff
        last_err: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                return _http_chat_completion(url, headers, payload, timeout)
            except Exception as e:
                last_err = e
                logger.warning("experience_summarizer teacher attempt %s failed: %s", attempt, e)
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
        assert last_err is not None
        raise last_err

    if max_concurrent <= 1:
        return [one(i, t) for i, t in enumerate(trajectory_user_messages)]

    out: List[Optional[str]] = [None] * len(trajectory_user_messages)
    with ThreadPoolExecutor(max_workers=min(max_concurrent, len(trajectory_user_messages))) as ex:
        futs = {ex.submit(one, i, trajectory_user_messages[i]): i for i in range(len(trajectory_user_messages))}
        for fut in as_completed(futs):
            idx = futs[fut]
            out[idx] = fut.result()
    return [x or "" for x in out]


def _summarize_batch_self(
    *,
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    config: DictConfig,
    es: DictConfig,
    actor_rollout_wg,
) -> List[str]:
    """Summarize trajectories using the actor rollout worker."""
    summ_max = OmegaConf.select(es, "summarizer_max_prompt_length", default=None)
    max_prompt_length = int(config.data.max_prompt_length) if summ_max is None else int(summ_max)
    summ_trunc = OmegaConf.select(es, "summarizer_prompt_truncation", default=None)
    truncation = str(config.data["truncation"]) if summ_trunc is None else str(summ_trunc)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    rows = [_row_from_prompt(tokenizer, p, max_prompt_length, int(pad_id), truncation) for p in prompts]
    data = collate_fn(rows)
    meta = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": int(pad_id),
        "recompute_log_prob": False,
        "do_sample": bool(es["do_sample"]),
        "validate": False,
        "response_length": int(es["max_new_tokens"]),
        "temperature": float(es["temperature"]),
        "top_p": float(es["top_p"]),
    }
    dp = DataProto.from_single_dict(data=data, meta_info=meta)
    pad_size = 0
    try:
        world_size = int(actor_rollout_wg.world_size)
    except Exception:
        world_size = 1
    dp_pad, pad_size = pad_dataproto_to_divisor(dp, world_size)
    out_pad = actor_rollout_wg.generate_sequences(dp_pad)
    out = unpad_dataproto(out_pad, pad_size=pad_size)
    n = len(prompts)
    texts = tokenizer.batch_decode(out.batch["responses"][:n], skip_special_tokens=True)
    return [t.strip() for t in texts]


def _chunk_ranges(length: int, chunk_size: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= length:
        return [(0, length)]
    return [(i, min(i + chunk_size, length)) for i in range(0, length, chunk_size)]


def _parse_memories_from_model_output(raw: str) -> list[dict[str, Any]]:
    try:
        obj = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("experience_summarizer: JSON parse failed: %s", e)
        return []
    memories = obj.get("memories")
    if not isinstance(memories, list):
        return []
    return [m for m in memories if isinstance(m, dict)]


def _memory_records_from_extractor_output(
    raw: str,
    *,
    pending: Tuple[int, str, float, bool, str, Any],
    task_name: str,
    mode: str,
    model_label: str,
    max_pairs: int,
    max_state_chars: int,
    max_action_chars: int,
    max_memory_chars: int,
) -> list[MemoryRecord]:
    """Parse one trajectory's model output and return records that pass field validation."""
    env_index, traj_s, r, succ, traj_plain, dataset_item_id = pending
    memories = _parse_memories_from_model_output(raw)[:max_pairs]
    out: list[MemoryRecord] = []
    for midx, mem_obj in enumerate(memories):
        rec = _memory_record_from_extracted_dict(
            memory=mem_obj,
            memory_idx=midx,
            task_name=task_name,
            trajectory_index=env_index,
            dataset_item_id=dataset_item_id,
            source_episode_id=traj_s,
            episode_reward=r,
            episode_success=succ,
            mode=mode,
            trajectory_char_len=len(traj_plain),
            extractor_model_label=model_label,
            max_state_chars=max_state_chars,
            max_action_chars=max_action_chars,
            max_memory_chars=max_memory_chars,
        )
        if rec is not None:
            out.append(rec)
    return out


def _run_extraction_inference_for_indices(
    indices: List[int],
    *,
    mode: str,
    trajectory_user_messages: List[str],
    chat_prompts: List[str] | None,
    summarization_batch_size: int,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    config: DictConfig,
    es: DictConfig,
    actor_rollout_wg,
    oa: DictConfig | None,
) -> List[str]:
    """Run teacher/self extraction for ``indices`` (order preserved); returns one raw string per index."""
    if not indices:
        return []
    if mode == "self":
        if chat_prompts is None:
            raise ValueError("chat_prompts required for mode=self")
        sub_prompts = [chat_prompts[i] for i in indices]
        texts: List[str] = []
        for start, end in _chunk_ranges(len(sub_prompts), summarization_batch_size or len(sub_prompts)):
            texts.extend(
                _summarize_batch_self(
                    prompts=sub_prompts[start:end],
                    tokenizer=tokenizer,
                    config=config,
                    es=es,
                    actor_rollout_wg=actor_rollout_wg,
                )
            )
        return texts
    if mode == "teacher":
        if oa is None:
            raise ValueError("openai_api config required for mode=teacher")
        sub_msgs = [trajectory_user_messages[i] for i in indices]
        texts = []
        for start, end in _chunk_ranges(len(sub_msgs), summarization_batch_size or len(sub_msgs)):
            texts.extend(
                _summarize_batch_teacher(
                    trajectory_user_messages=sub_msgs[start:end],
                    system_prompt=system_prompt,
                    oa=oa,
                )
            )
        return texts
    raise ValueError(f"unsupported mode={mode}")


def _memory_record_from_extracted_dict(
    *,
    memory: dict[str, Any],
    memory_idx: int,
    task_name: str,
    trajectory_index: int,
    dataset_item_id: Any,
    source_episode_id: str,
    episode_reward: float,
    episode_success: bool,
    mode: str,
    trajectory_char_len: int,
    extractor_model_label: str,
    max_state_chars: int,
    max_action_chars: int,
    max_memory_chars: int,
) -> MemoryRecord | None:
    state_text = normalize_text(str(memory.get("state_text") or ""), max_chars=max_state_chars)
    action_text = normalize_text(str(memory.get("action_text") or ""), max_chars=max_action_chars)
    memory_text = normalize_text(str(memory.get("memory_text") or ""), max_chars=max_memory_chars)
    if not state_text or not action_text or not memory_text:
        return None

    metadata = memory.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    metadata = {
        **metadata,
        "dataset_item_id": dataset_item_id,
        "trajectory_index": trajectory_index,
        "memory_index_within_trajectory": memory_idx,
        "extracted_by": extractor_model_label,
    }

    source_step = memory.get("source_step", memory_idx + 1)
    try:
        source_step = int(source_step)
    except (TypeError, ValueError):
        source_step = memory_idx + 1

    value = memory.get("value", None)
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None

    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return MemoryRecord(
        memory_id=str(uuid.uuid4()),
        task_name=task_name,
        item_id=int(trajectory_index),
        source_episode_id=source_episode_id,
        source_step=source_step,
        state_text=state_text,
        action_text=action_text,
        memory_text=memory_text,
        reward=float(episode_reward),
        success=bool(episode_success),
        created_step=None,
        created_at=now_ts,
        retrieval_count=0,
        last_used_step=None,
        metadata={
            **metadata,
            "source": "experience_summarizer",
            "mode": mode,
            "trajectory_chars": trajectory_char_len,
        },
        value=value,
        value_source="llm_extraction" if value is not None else None,
        value_update_step=None,
    )


def maybe_summarize_and_write_experiences(
    *,
    config: DictConfig,
    memory_manager: MemoryManager,
    tokenizer: PreTrainedTokenizer,
    actor_rollout_wg,
    total_batch_list: List[List[Dict[str, Any]]],
    total_infos: List[List[Dict[str, Any]]],
    episode_rewards: np.ndarray,
    episode_lengths: np.ndarray,
    success: Dict[str, np.ndarray],
    traj_uid: np.ndarray,
) -> int:
    """
    Implementation for ``MemoryManager.maybe_write_rollout_memories`` (also usable directly in tests).

    Requires ``env.memory.enabled``, ``env.memory.write_back``, and
    ``env.memory.experience_summarizer.mode`` in {``self``, ``teacher``}.
    The model response must be parseable as JSON with a ``memories`` array (see ``scripts/extract_memory_records.py``).
    If parsing yields no memories or every item fails validation, the same trajectory is inferred again until
    ``parse_max_attempts`` is reached (per trajectory).
    """
    if not experience_summarizer_active(config):
        return 0
    rm = memory_manager
    if not rm.enabled or rm.store is None:
        logger.debug("experience_summarizer: memory store unavailable; skip.")
        return 0

    mem = config.env.memory
    es = _exps_cfg(config)
    mode = str(es["mode"]).strip().lower()
    only_pos = bool(mem.only_store_positive_reward)
    max_pairs = max(1, int(mem.max_pairs_per_episode))
    max_traj_chars = int(es["max_trajectory_chars"])
    num_mem_cfg = int(es["num_memories_per_trajectory"])
    num_memories_cap = max(1, min(num_mem_cfg, max_pairs))
    summarization_batch_size = int(es["summarization_batch_size"])
    system_prompt = _resolve_system_prompt(es)
    trajectory_user_prompt_template = _resolve_trajectory_user_prompt_template(es)
    if not trajectory_user_prompt_template.strip():
        trajectory_user_prompt_template = DEFAULT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE
    apply_kw = OmegaConf.to_container(config.data["apply_chat_template_kwargs"], resolve=True) or {}

    batch_size = len(total_batch_list)
    trajectory_user_messages: List[str] = []
    pending_rollouts: List[Tuple[int, str, float, bool, str, Any]] = []

    for i in range(batch_size):
        steps = total_batch_list[i]
        if not steps:
            continue
        r = float(np.asarray(episode_rewards[i], dtype=np.float64).reshape(-1)[0])
        if only_pos and r <= 0.0:
            continue
        succ = _episode_success(success, i)
        traj_s = str(traj_uid[i])
        traj_plain = _format_trajectory_plain(steps, tokenizer, max_chars=max_traj_chars)
        if not traj_plain.strip():
            continue
        dataset_item_id = _dataset_item_id_from_infos(total_infos, i)
        ep_len = float(episode_lengths[i]) if i < len(episode_lengths) else 0.0
        try:
            user_message = trajectory_user_prompt_template.format(
                num_memories=num_memories_cap,
                task_name=rm.task_name,
                dataset_item_id=dataset_item_id,
                trajectory_index=int(i),
                trajectory_text=traj_plain,
                episode_reward=r,
                episode_length=ep_len,
                traj_uid=traj_s,
                success=int(succ),
            )
        except KeyError as e:
            logger.warning("experience_summarizer: template placeholder missing: %s; skip env %s.", e, i)
            continue
        trajectory_user_messages.append(user_message)
        pending_rollouts.append((i, traj_s, r, succ, traj_plain, dataset_item_id))

    if not trajectory_user_messages:
        return 0
    if mode not in ("self", "teacher"):
        raise ValueError(f"unsupported mode={mode}")

    model_label = str(es["extractor_model_label"])
    oa: DictConfig | None = None
    if mode == "teacher":
        oa = es["openai_api"]
        model_label = str(oa["model"])

    parse_max_attempts = max(1, int(es["parse_max_attempts"]))

    n_pending = len(trajectory_user_messages)
    raw_outputs: List[str] = [""] * n_pending
    parse_attempts: List[int] = [0] * n_pending
    active: List[int] = list(range(n_pending))

    chat_prompts: List[str] | None = None
    if mode == "self":
        if actor_rollout_wg is None:
            logger.warning("experience_summarizer mode=self but actor_rollout_wg is None; skip.")
            return 0
        chat_prompts = [
            _build_chat_prompt(
                tokenizer, system_prompt, user_msg, apply_kw if isinstance(apply_kw, dict) else None
            )
            for user_msg in trajectory_user_messages
        ]

    max_state = int(mem.max_state_chars)
    max_act = int(mem.max_action_chars)
    max_mem = int(mem.max_memory_chars)

    try:
        while active:
            batch_out = _run_extraction_inference_for_indices(
                active,
                mode=mode,
                trajectory_user_messages=trajectory_user_messages,
                chat_prompts=chat_prompts,
                summarization_batch_size=summarization_batch_size,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                config=config,
                es=es,
                actor_rollout_wg=actor_rollout_wg,
                oa=oa,
            )
            for j, idx in enumerate(active):
                raw_outputs[idx] = batch_out[j]
                parse_attempts[idx] += 1

            next_active: List[int] = []
            for idx in active:
                recs = _memory_records_from_extractor_output(
                    raw_outputs[idx],
                    pending=pending_rollouts[idx],
                    task_name=rm.task_name,
                    mode=mode,
                    model_label=model_label,
                    max_pairs=max_pairs,
                    max_state_chars=max_state,
                    max_action_chars=max_act,
                    max_memory_chars=max_mem,
                )
                if recs:
                    continue
                if parse_attempts[idx] >= parse_max_attempts:
                    logger.warning(
                        "experience_summarizer: trajectory index %s (env %s) still has no valid memories "
                        "after %s attempt(s); giving up.",
                        idx,
                        pending_rollouts[idx][0],
                        parse_attempts[idx],
                    )
                    continue
                next_active.append(idx)

            if next_active:
                logger.info(
                    "experience_summarizer: re-running extraction for %s trajectory(s) with empty/invalid parse "
                    "(attempt cap %s per trajectory).",
                    len(next_active),
                    parse_max_attempts,
                )
            active = next_active
    except Exception:
        logger.exception("experience_summarizer failed during model calls; no records written.")
        return 0

    records: List[MemoryRecord] = []
    for idx in range(n_pending):
        records.extend(
            _memory_records_from_extractor_output(
                raw_outputs[idx],
                pending=pending_rollouts[idx],
                task_name=rm.task_name,
                mode=mode,
                model_label=model_label,
                max_pairs=max_pairs,
                max_state_chars=max_state,
                max_action_chars=max_act,
                max_memory_chars=max_mem,
            )
        )

    if not records:
        return 0
    try:
        rm.store.add_records(records)
        logger.info("experience_summarizer wrote %s memory record(s) (mode=%s).", len(records), mode)
    except Exception:
        logger.exception("experience_summarizer failed to add_records.")
        return 0
    return len(records)
