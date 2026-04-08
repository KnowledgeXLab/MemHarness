# Copyright 2025 MemAdaptor
#
# Online experience / memory write-back: summarize full trajectories then insert into the memory VDB.
# - mode=self: batch prompts to ``actor_rollout_wg.generate_sequences`` (same Reasoning rollout worker).
# - mode=teacher: OpenAI-compatible ``/v1/chat/completions`` on the driver (no Ray worker).

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

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
        except Exception:  # noqa: BLE001
            return bool(a[index])
    return False


def _build_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    user_body: str,
    apply_chat_template_kwargs: Optional[dict],
) -> str:
    apply_kw = dict(apply_chat_template_kwargs or {})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_body},
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
    user_contents: List[str],
    system_prompt: str,
    oa: DictConfig,
) -> List[str]:
    base_url = str(oa.get("base_url", "https://api.openai.com/v1"))
    url = _normalize_chat_url(base_url)
    api_key = str(oa.get("api_key", "") or "")
    if not api_key:
        raise ValueError(
            "env.memory.experience_summarizer.openai_api.api_key is empty; set it or OPENAI_API_KEY."
        )
    model = str(oa.get("model", "gpt-4o-mini"))
    timeout = float(oa.get("timeout_sec", 120.0))
    max_tokens = int(oa.get("max_tokens", 512))
    temperature = float(oa.get("temperature", 0.3))
    max_concurrent = int(oa.get("max_concurrent", 8))
    max_retries = int(oa.get("max_retries", 2))
    retry_backoff = float(oa.get("retry_backoff_sec", 1.0))
    extra = oa.get("extra_headers")
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

        delay = retry_backoff
        last_err: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                return _http_chat_completion(url, headers, payload, timeout)
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("experience_summarizer teacher attempt %s failed: %s", attempt, e)
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
        assert last_err is not None
        raise last_err

    if max_concurrent <= 1:
        return [one(i, t) for i, t in enumerate(user_contents)]

    out: List[Optional[str]] = [None] * len(user_contents)
    with ThreadPoolExecutor(max_workers=min(max_concurrent, len(user_contents))) as ex:
        futs = {ex.submit(one, i, user_contents[i]): i for i in range(len(user_contents))}
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
    max_prompt_length = int(config.data.max_prompt_length)
    truncation = str(config.data.get("truncation", "error"))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    rows = [_row_from_prompt(tokenizer, p, max_prompt_length, int(pad_id), truncation) for p in prompts]
    data = collate_fn(rows)
    meta = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": int(pad_id),
        "recompute_log_prob": False,
        "do_sample": bool(es.get("do_sample", False)),
        "validate": False,
        "response_length": int(es.get("max_new_tokens", 512)),
        "temperature": float(es.get("temperature", 0.3)),
        "top_p": float(es.get("top_p", 1.0)),
    }
    dp = DataProto.from_single_dict(data=data, meta_info=meta)
    pad_size = 0
    try:
        world_size = int(actor_rollout_wg.world_size)
    except Exception:  # noqa: BLE001
        world_size = 1
    dp_pad, pad_size = pad_dataproto_to_divisor(dp, world_size)
    out_pad = actor_rollout_wg.generate_sequences(dp_pad)
    out = unpad_dataproto(out_pad, pad_size=pad_size)
    n = len(prompts)
    texts = tokenizer.batch_decode(out.batch["responses"][:n], skip_special_tokens=True)
    return [t.strip() for t in texts]


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
    """
    if not experience_summarizer_active(config):
        return 0
    rm = memory_manager
    if not rm.enabled or rm.store is None:
        logger.debug("experience_summarizer: memory store unavailable; skip.")
        return 0

    mem = config.env.memory
    es = _exps_cfg(config)
    mode = str(es.get("mode", "none")).strip().lower()
    only_pos = bool(mem.only_store_positive_reward)
    max_pairs = max(1, int(mem.max_pairs_per_episode))
    max_traj_chars = int(es.get("max_trajectory_chars", 32000))
    system_prompt = str(es.get("system_prompt", "")).strip()
    user_tmpl = str(es.get("user_template", "{trajectory_text}"))
    apply_kw = OmegaConf.to_container(config.data.get("apply_chat_template_kwargs", {}), resolve=True) or {}

    records: List[MemoryRecord] = []

    batch_size = len(total_batch_list)
    user_bodies: List[str] = []
    meta_rows: List[tuple] = []

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
        ub = user_tmpl.format(
            trajectory_text=traj_plain,
            episode_reward=r,
            episode_length=float(episode_lengths[i]) if i < len(episode_lengths) else 0.0,
            traj_uid=traj_s,
            success=int(succ),
        )
        if max_pairs < 1:
            continue
        user_bodies.append(ub)
        meta_rows.append((i, traj_s, r, succ, traj_plain))

    if not user_bodies:
        return 0

    try:
        if mode == "self":
            if actor_rollout_wg is None:
                logger.warning("experience_summarizer mode=self but actor_rollout_wg is None; skip.")
                return 0
            chat_prompts = [
                _build_chat_prompt(tokenizer, system_prompt, ub, apply_kw if isinstance(apply_kw, dict) else None)
                for ub in user_bodies
            ]
            summaries = _summarize_batch_self(
                prompts=chat_prompts,
                tokenizer=tokenizer,
                config=config,
                es=es,
                actor_rollout_wg=actor_rollout_wg,
            )
        elif mode == "teacher":
            oa = es.get("openai_api") or OmegaConf.create({})
            summaries = _summarize_batch_teacher(user_contents=user_bodies, system_prompt=system_prompt, oa=oa)
        else:
            return 0
    except Exception:  # noqa: BLE001
        logger.exception("experience_summarizer failed during model calls; no records written.")
        return 0

    max_state = int(mem.max_state_chars)
    max_mem = int(mem.max_memory_chars)

    for (i, traj_s, r, succ, traj_plain), summary in zip(meta_rows, summaries):
        summary = normalize_text(summary, max_chars=max_mem)
        if not summary:
            continue
        steps_i = total_batch_list[i]
        first_state = _state_text_from_step(steps_i[0], tokenizer) if steps_i else ""
        first_state = normalize_text(first_state, max_chars=max_state)

        records.append(
            MemoryRecord(
                memory_id=str(uuid.uuid4()),
                task_name=rm.task_name,
                item_id=int(i),
                source_episode_id=traj_s,
                source_step=0,
                state_text=first_state,
                action_text="",
                memory_text=summary,
                reward=float(r),
                success=bool(succ),
                metadata={
                    "source": "experience_summarizer",
                    "mode": mode,
                    "trajectory_chars": len(traj_plain),
                },
            )
        )

    if not records:
        return 0
    try:
        rm.store.add_records(records)
        logger.info("experience_summarizer wrote %s memory record(s) (mode=%s).", len(records), mode)
    except Exception:  # noqa: BLE001
        logger.exception("experience_summarizer failed to add_records.")
        return 0
    return len(records)
