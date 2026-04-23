# Copyright 2025 MemAdaptor
#
# Memory Adaptor: inference hook (P_new / <EMPTY>) batched via WorkerGroup.generate_sequences.
# Training: see ``mem_adaptor_training`` + ``train_memory_adaptor`` (GRPO; Reasoning optional frozen or joint).
# Rollout contract: ``verl.protocol.MEMORY_ADAPTOR_INFOS_KEYS`` / ``write_mem_adaptor_step_non_tensor_batch``.

from __future__ import annotations

from collections import defaultdict
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from agent_system.memory.global_step_schedule import match_phase_for_global_step


def _mem_cfg(config: DictConfig) -> DictConfig:
    ma = OmegaConf.select(config, "mem_adaptor")
    return ma if ma is not None else OmegaConf.create({})


def _plain_dict_chat_kw(obj: Any) -> dict:
    """OmegaConf ``to_container`` rejects plain dicts; runtime may already resolve to dict."""
    if obj is None:
        return {}
    if OmegaConf.is_config(obj):
        return dict(OmegaConf.to_container(obj, resolve=True) or {})
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def adaptor_apply_chat_template_kwargs(config: DictConfig, ma: Optional[DictConfig] = None) -> dict:
    """Kwargs passed to ``tokenizer.apply_chat_template`` when building adaptor prompts.

    If ``mem_adaptor.apply_chat_template_kwargs`` is **non-null**, it wins; otherwise
    ``data.apply_chat_template_kwargs`` is used. The **adaptor** ``PreTrainedTokenizer``
    (``mem_adaptor.actor_rollout_ref.model.path``) must match this template (same model
    family / instruct format as intended); mismatch can encourage garbage completions.
    """
    ma = ma if ma is not None else _mem_cfg(config)
    ma_kw = OmegaConf.select(ma, "apply_chat_template_kwargs", default=None)
    if ma_kw is not None:
        return _plain_dict_chat_kw(ma_kw)
    data_kw = OmegaConf.select(config, "data.apply_chat_template_kwargs", default={}) or {}
    return _plain_dict_chat_kw(data_kw)


def mem_adaptor_enabled(config: DictConfig) -> bool:
    ma = _mem_cfg(config)
    return bool(ma.get("enable", False))


def _normalize_adaptor_output(text: str, empty_markers: Sequence[str]) -> Tuple[str, bool]:
    """Returns (stripped text, is_empty_reject)."""
    t = text.strip()
    low = t.lower()
    for m in empty_markers:
        if t == m or low == m.lower():
            return "", True
    return t, False


def _retrieved_state_principle_pairs(memory_event: Any, top_k: int) -> List[Tuple[str, str]]:
    """S_old, P_old pairs from MemoryEvent.to_dict()['retrieved'][:top_k]."""
    if not memory_event or not isinstance(memory_event, dict) or top_k <= 0:
        return []
    ret = memory_event["retrieved"] or []
    out: List[Tuple[str, str]] = []
    for r in ret[:top_k]:
        if not isinstance(r, dict):
            continue
        out.append(
            (
                str(r["state_text"]),
                str(r["memory_text"]),
            )
        )
    return out


def _top1_retrieved_fields(memory_event: Any) -> Tuple[str, str]:
    pairs = _retrieved_state_principle_pairs(memory_event, 1)
    return pairs[0] if pairs else ("", "")


def _should_run_adaptor_for_index(
    schedule: str,
    info: Dict[str, Any],
    *,
    active: bool,
) -> bool:
    if not active:
        return False
    if schedule == "every_step":
        return True
    # if the memory is injected, return True
    injected = (info["memory_injected_text"] or "").strip()
    ev = info["memory_event"]
    # if the memory event has retrieved, return True
    has_ret = bool(isinstance(ev, dict) and (ev["retrieved"] or []))
    return bool(injected) or has_ret


def _passes_train_global_window(ma: DictConfig, trainer_global_step: Optional[int]) -> bool:
    """If ``trainer_global_step`` is None, no global bound is applied."""
    if trainer_global_step is None:
        return True
    t0 = ma["train_global_step_start"]
    t1 = ma["train_global_step_end"]
    if t0 is not None and trainer_global_step < int(t0):
        return False
    if t1 is not None and trainer_global_step >= int(t1):
        return False
    return True


def _passes_env_step_window(
    ma: DictConfig,
    env_step_1based: Optional[int],
    trainer_global_step: Optional[int] = None,
) -> bool:
    """Env step index after the current env step (matches post-increment ``episode_lengths[i]``).

    When ``mem_adaptor.env_step_phases`` is set, the first matching global-step phase overrides
    ``env_step_start`` / ``env_step_end`` / ``env_step_every_n`` for that window.
    """
    if env_step_1based is None:
        return True
    phase = match_phase_for_global_step(ma.get("env_step_phases"), trainer_global_step)
    es0 = ma["env_step_start"]
    es1 = ma["env_step_end"]
    ev_n = ma["env_step_every_n"]
    if phase:
        if phase.get("env_step_start") is not None:
            es0 = phase["env_step_start"]
        if phase.get("env_step_end") is not None:
            es1 = phase["env_step_end"]
        if phase.get("env_step_every_n") is not None:
            ev_n = phase["env_step_every_n"]
    ev_n = int(ev_n)
    start = int(es0) if es0 is not None else 1
    if env_step_1based < start:
        return False
    if es1 is not None and env_step_1based >= int(es1):
        return False
    if ev_n > 1 and (env_step_1based - start) % ev_n != 0:
        return False
    return True


def _clip(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _build_adaptor_prompt(
    tokenizer: PreTrainedTokenizer,
    ma: DictConfig,
    *,
    s_curr: str,
    s_old: str,
    p_old: str,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> str:
    tmpl = str(ma["user_message_template"])
    user_content = tmpl.format(
        s_curr=_clip(s_curr, int(ma["max_prompt_chars"])),
        s_old=_clip(s_old, int(ma["max_prompt_chars"])),
        p_old=_clip(p_old, int(ma["max_prompt_chars"])),
    )
    system_prompt = str(ma["system_prompt"])
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


def _row_dict_from_prompt(
    *,
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


def build_adaptor_dataproto(
    *,
    config: DictConfig,
    tokenizer: PreTrainedTokenizer,
    rows: List[dict],
    meta_extras: Optional[dict] = None,
) -> DataProto:
    """Collate adaptor prefill rows into DataProto with meta_info for generate_sequences."""
    if not rows:
        raise ValueError("build_adaptor_dataproto: empty rows")
    data = collate_fn(rows)
    ma = _mem_cfg(config)
    pad_id = tokenizer.pad_token_id
    meta = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": pad_id if pad_id is not None else tokenizer.eos_token_id,
        "recompute_log_prob": False,
        "do_sample": bool(ma.get("do_sample", True)),
        "validate": False,
        "response_length": int(ma.get("max_new_tokens", 256)),
        "temperature": float(ma.get("temperature", 0.7)),
        "top_p": float(ma.get("top_p", 1.0)),
    }
    if meta_extras:
        meta.update(meta_extras)
    return DataProto.from_single_dict(data=data, meta_info=meta)


def _replacement_from_many_raws(raw_outputs: Sequence[str], ma: DictConfig) -> Tuple[str, bool]:
    """Build merged replacement string and whether every hit is a reject (<EMPTY>).

    When ``adapted_principle_label_template`` is set (non-empty), each accepted output becomes
    ``template`` with ``{index}`` -> 1-based hit index, then bodies are joined with ``multi_hit_joiner``.
    Otherwise falls back to a single ``adapted_injection_prefix`` before joined bodies (legacy).
    """
    markers = list(ma["empty_output_markers"])
    no_exp = str(ma["no_experience_message"]).strip()
    joiner = str(ma["multi_hit_joiner"])
    label_tmpl_raw = ma.get("adapted_principle_label_template")

    cleaned_parts: List[str] = []
    any_accept = False
    for raw in raw_outputs:
        cleaned, is_reject = _normalize_adaptor_output(str(raw), markers)
        if not is_reject and cleaned:
            cleaned_parts.append(cleaned)
            any_accept = True

    if not any_accept:
        # Observation patching uses all_reject to strip <memory>...</memory>; keep message for config/docs only.
        rep = no_exp if no_exp else ""
        return rep, True

    if label_tmpl_raw is not None and str(label_tmpl_raw).strip():
        tmpl = str(label_tmpl_raw)
        blocks: List[str] = []
        for idx, part in enumerate(cleaned_parts):
            lab = tmpl.replace("{index}", str(idx + 1))
            blocks.append(f"{lab}{part.strip()}")
        rep = joiner.join(blocks)
        return rep, False

    prefix = str(ma.get("adapted_injection_prefix", "") or "")
    body = joiner.join(cleaned_parts)
    if prefix and body:
        rep = f"{prefix}{body}"
    elif prefix:
        rep = prefix.rstrip()
    else:
        rep = body
    return rep, False


_MEMORY_BLOCK_RE = re.compile(r"<memory>\s*.*?\s*</memory>", re.DOTALL | re.IGNORECASE)


def _splice_out_memory_block(text: str, injected_inner: str, new_inner: Optional[str]) -> str:
    """Replace the last ``<memory>...</memory>`` block.

    ``new_inner`` None or empty string => remove the block (and the boilerplate header/memories inside it).
    Otherwise set inner content to ``new_inner`` (no extra wrapping).
    """
    t = text or ""
    span: Optional[tuple[int, int]] = None
    matches = list(_MEMORY_BLOCK_RE.finditer(t))
    if matches:
        span = matches[-1].span()
    elif injected_inner:
        for candidate in (
            f"<memory>\n{injected_inner}\n</memory>",
            f"<memory>\n{injected_inner.strip()}\n</memory>",
        ):
            s = t.rfind(candidate)
            if s != -1:
                span = (s, s + len(candidate))
                break
    if span is None:
        return t
    s, e = span
    before = t[:s].rstrip()
    after = t[e:].lstrip()
    inner = (new_inner or "").strip()
    if not inner:
        parts = [p for p in (before, after) if p]
        return "\n\n".join(parts) if parts else ""
    block = f"<memory>\n{inner}\n</memory>"
    parts = [p for p in (before, block, after) if p]
    return "\n\n".join(parts)


def _patch_one_obs_text(
    *,
    texts: List[str],
    i: int,
    infos: List[Dict[str, Any]],
    replacement: str,
    raw_outputs: Sequence[str],
    all_reject: bool,
) -> None:
    """Replace the last ``<memory>...</memory>`` inner with adaptor output; on <EMPTY> remove the block and append ``replacement`` (``no_experience_message``)."""

    infos[i]["mem_adaptor_raw_outputs"] = list(raw_outputs)
    infos[i]["mem_adaptor_raw_output"] = raw_outputs[0]
    infos[i]["mem_adaptor_reject"] = all_reject

    injected = infos[i]["memory_injected_text"].strip()
    t = texts[i] or ""

    if all_reject:
        new_t = _splice_out_memory_block(t, injected, None)
        no_exp = (replacement or "").strip()
        if no_exp:
            new_t = new_t.rstrip() + "\n\n" + no_exp
        texts[i] = new_t
        infos[i]["mem_adaptor_applied"] = True
        infos[i]["memory_injected_text"] = no_exp if no_exp else ""
        return

    rep = (replacement or "").strip()
    if not rep:
        texts[i] = _splice_out_memory_block(t, injected, None)
        infos[i]["mem_adaptor_applied"] = True
        infos[i]["memory_injected_text"] = ""
        return

    if not injected and "<memory>" not in t.lower():
        texts[i] = t.rstrip() + f"\n\n<memory>\n{rep}\n</memory>"
        infos[i]["mem_adaptor_applied"] = True
        infos[i]["memory_injected_text"] = rep
        return

    texts[i] = _splice_out_memory_block(t, injected, rep)
    infos[i]["mem_adaptor_applied"] = True
    infos[i]["memory_injected_text"] = rep


def apply_adaptor_to_obs_texts(
    *,
    texts: List[str],
    infos: List[Dict[str, Any]],
    indices: List[int],
    adaptor_texts: List[str],
    ma: DictConfig,
) -> None:
    """In-place patch ``texts[i]`` after retrieval block; set ``infos[i]`` adaptor fields."""
    for row_k, i in enumerate(indices):
        raw = adaptor_texts[row_k]
        rep, reject = _replacement_from_many_raws([raw], ma)
        _patch_one_obs_text(texts=texts, i=i, infos=infos, replacement=rep, raw_outputs=[raw], all_reject=reject)


def maybe_apply_memory_adaptor(
    *,
    config: DictConfig,
    tokenizer: PreTrainedTokenizer,
    next_obs: Dict[str, Any],
    infos: List[Dict[str, Any]],
    active_masks: np.ndarray,
    generate_wg,
    adaptor_rollout_wg=None,
    trainer_global_step: Optional[int] = None,
    episode_lengths: Optional[np.ndarray] = None,
    adaptor_training_buffer: Optional[List[Dict[str, Any]]] = None,
    traj_uid: Optional[np.ndarray] = None,
    grpo_group_uid: Optional[np.ndarray] = None,
) -> None:
    """
    After ``step_with_memory``, optionally run batched Adaptor ``generate_sequences`` and
    patch ``next_obs['text']``. Mutates ``next_obs`` and ``infos`` in place.

    For training, pass ``grpo_group_uid`` (same layout as ``traj_uid``): Reasoning GRPO group id
    per env, reused across ``env.rollout.n`` parallel envs so Adaptor GRPO can compare trajectory
    returns within the same group (see ``mem_adaptor_training``).
    """
    if not mem_adaptor_enabled(config):
        return

    ma = _mem_cfg(config)
    schedule = str(ma["schedule"])
    top_k = max(1, int(ma["retrieval_top_k"]))
    wg = adaptor_rollout_wg
    if wg is None and bool(ma["use_actor_rollout_wg"]):
        wg = generate_wg
    if wg is None:
        return

    if not _passes_train_global_window(ma, trainer_global_step):
        return

    texts = next_obs["text"]
    if not isinstance(texts, list):
        return

    anchors = next_obs["anchor"]
    batch_n = len(texts)
    max_prompt_length = int(config.data["max_prompt_length"])
    truncation = str(config.data["truncation"])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    chat_kw = adaptor_apply_chat_template_kwargs(config, ma)

    row_env_indices: List[int] = []
    rows: List[dict] = []
    for i in range(batch_n):
        if not _should_run_adaptor_for_index(schedule, infos[i], active=bool(active_masks[i])):
            continue
        env_step_i: Optional[int] = None
        if episode_lengths is not None and i < len(episode_lengths):
            env_step_i = int(episode_lengths[i])
        if not _passes_env_step_window(ma, env_step_i, trainer_global_step):
            continue
        ev = infos[i]["memory_event"]
        pairs = _retrieved_state_principle_pairs(ev, top_k)
        if anchors is not None and i < len(anchors):
            s_curr = str(anchors[i] if anchors[i] is not None else "")
        else:
            s_curr = str(texts[i] if texts[i] is not None else "")

        if schedule == "on_memory_only":
            injected = infos[i]["memory_injected_text"].strip()
            if not pairs and not injected:
                continue
            if not pairs and injected:
                # memory injected without structured retrieved list — single adaptor call with empty S_old/P_old
                pairs = [("", "")]

        if not pairs and schedule == "every_step":
            ph_s = str(ma["placeholder_old_state"])
            ph_p = str(ma["placeholder_old_principle"])
            pairs = [(ph_s, ph_p)]

        for s_old, p_old in pairs:
            if schedule == "on_memory_only" and not (s_old or p_old) and not infos[i]["memory_injected_text"].strip():
                continue
            if not s_old and not p_old and schedule == "every_step":
                s_old = str(ma["placeholder_old_state"])
                p_old = str(ma["placeholder_old_principle"])

            prompt = _build_adaptor_prompt(
                tokenizer,
                ma,
                s_curr=s_curr,
                s_old=s_old,
                p_old=p_old,
                apply_chat_template_kwargs=chat_kw,
            )
            row = _row_dict_from_prompt(
                tokenizer=tokenizer,
                prompt=prompt,
                max_prompt_length=max_prompt_length,
                pad_token_id=int(pad_token_id),
                truncation=truncation,
            )
            row_env_indices.append(i)
            rows.append(row)

    if not rows:
        return

    dp = build_adaptor_dataproto(config=config, tokenizer=tokenizer, rows=rows)
    pad_size = 0
    try:
        world_size = int(wg.world_size)
    except Exception:
        world_size = 1
    dp_pad, pad_size = pad_dataproto_to_divisor(dp, world_size)
    out_pad = wg.generate_sequences(dp_pad)
    out = unpad_dataproto(out_pad, pad_size=pad_size)
    n = len(row_env_indices)
    dec = tokenizer.batch_decode(out.batch["responses"][:n], skip_special_tokens=True)

    if adaptor_training_buffer is not None and n > 0 and traj_uid is not None:
        pad_tok = int(pad_token_id)
        empty_markers = list(ma["empty_output_markers"])
        for k in range(n):
            env_i = row_env_indices[k]
            if env_i < 0 or env_i >= len(traj_uid):
                continue
            if grpo_group_uid is not None and env_i < len(grpo_group_uid):
                grpo_id = str(grpo_group_uid[env_i])
            else:
                grpo_id = str(uuid.uuid4())
            _, call_reject = _normalize_adaptor_output(str(dec[k]), empty_markers)
            rec: Dict[str, Any] = {
                "prompts": out.batch["prompts"][k].detach().cpu().clone(),
                "responses": out.batch["responses"][k].detach().cpu().clone(),
                "input_ids": out.batch["input_ids"][k].detach().cpu().clone(),
                "attention_mask": out.batch["attention_mask"][k].detach().cpu().clone(),
                "position_ids": out.batch["position_ids"][k].detach().cpu().clone(),
                "traj_uid": str(traj_uid[env_i]),
                "grpo_index": grpo_id,
                "pad_token_id": pad_tok,
                "mem_adaptor_reject": bool(call_reject),
            }
            if "rollout_log_probs" in out.batch.keys():
                rec["rollout_log_probs"] = out.batch["rollout_log_probs"][k].detach().cpu().clone()
            adaptor_training_buffer.append(rec)

    grouped: Dict[int, List[str]] = defaultdict(list)
    for env_i, line in zip(row_env_indices, dec):
        grouped[env_i].append(line)
    for env_i in sorted(grouped.keys()):
        raws = grouped[env_i]
        rep, reject = _replacement_from_many_raws(raws, ma)
        _patch_one_obs_text(
            texts=texts,
            i=env_i,
            infos=infos,
            replacement=rep,
            raw_outputs=raws,
            all_reject=reject,
        )
