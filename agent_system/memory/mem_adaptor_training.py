# Copyright 2025 MemAdaptor
#
# Train the Memory Adaptor policy with trajectory-level rewards (GRPO).
# Requires ``use_actor_rollout_wg=false`` (dedicated adaptor WorkerGroup). Reasoning may be frozen
# (``actor_rollout_ref.actor.trainable=false``) or trained in the same step (joint: leave Reasoning trainable).

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from collections import Counter

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.dataset.rl_dataset import collate_fn

# Default: penalize CJK, Japanese kana, Hangul, Cyrillic, Arabic (non–English-only completions for AlfWorld-style English tasks).
_GRPO_ENGLISH_SHAPING_DEFAULT_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]"
)


def _decode_adaptor_response_row(tokenizer: "PreTrainedTokenizer", resp: torch.Tensor, pad_id: int) -> str:
    row = resp.long().detach().cpu()
    mask = row != int(pad_id)
    if not mask.any():
        return ""
    try:
        return tokenizer.decode(row[mask], skip_special_tokens=True).strip()
    except Exception:
        return ""


def _mem_adaptor_grpo_english_shaping_delta(ma: DictConfig, response_text: str) -> tuple[float, bool]:
    """Return (scalar delta added to outcome reward, whether penalty applied)."""
    gs = OmegaConf.select(ma, "grpo_english_shaping", default=None)
    if gs is None or not bool(gs.get("enable", False)):
        return 0.0, False
    text = (response_text or "").strip()
    if not text:
        return 0.0, False
    pat_s = gs.get("pattern")
    if pat_s is not None and str(pat_s).strip():
        try:
            pat = re.compile(str(pat_s), re.UNICODE)
        except re.error:
            pat = _GRPO_ENGLISH_SHAPING_DEFAULT_RE
    else:
        pat = _GRPO_ENGLISH_SHAPING_DEFAULT_RE
    if pat.search(text):
        penalty = float(gs.get("penalty", 0.5))
        return -abs(penalty), True
    return 0.0, False


def scale_adaptor_grpo_advantages_by_traj_adaptor_steps(batch: DataProto, enabled: bool) -> None:
    """
    After ``compute_grpo_outcome_advantage``, optionally divide each row's advantages/returns by
    the number of Adaptor training rows from the same ``traj_uid`` (credit split across calls).
    """
    if not enabled or batch.batch is None or "advantages" not in batch.batch:
        return
    tu = batch.non_tensor_batch.get("traj_uid")
    if tu is None:
        return
    n_rows = len(tu)
    keys = [str(tu[i]) for i in range(n_rows)]
    counts = Counter(keys)
    adv = batch.batch["advantages"]
    for i in range(n_rows):
        c = counts[keys[i]]
        if c > 0:
            adv[i] = adv[i] / float(c)
    if "returns" in batch.batch:
        ret = batch.batch["returns"]
        for i in range(n_rows):
            c = counts[keys[i]]
            if c > 0:
                ret[i] = ret[i] / float(c)


def prune_mem_adaptor_training_samples_after_group_filter(
    buf: List[Dict[str, Any]],
    buf_start: int,
    kept_traj_uid: np.ndarray,
) -> None:
    """Keep only adaptor rows from the current inner rollout whose ``traj_uid`` survived ``filter_group_data``."""
    if buf_start < 0 or buf_start > len(buf):
        return
    new_chunk = buf[buf_start:]
    del buf[buf_start:]
    keep = {str(x) for x in np.asarray(kept_traj_uid, dtype=object).reshape(-1)}
    buf.extend(s for s in new_chunk if str(s.get("traj_uid", "")) in keep)


def format_mem_adaptor_training_sample_for_log(
    sample: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    *,
    max_prompt_chars: int = 4096,
    max_response_chars: int = 4096,
) -> str:
    """Decode one adaptor GRPO training row (CPU prompt/response ids) for stdout / file logs."""

    def _decode_ids(ids: Any) -> str:
        if ids is None:
            return ""
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu()
        try:
            return tokenizer.decode(ids, skip_special_tokens=True).strip()
        except Exception:
            return "<decode_error>"

    tu = str(sample.get("traj_uid", ""))
    gi = str(sample.get("grpo_index", ""))
    prompt = _decode_ids(sample.get("prompts"))
    resp = _decode_ids(sample.get("responses"))
    raw_pl, raw_rl = len(prompt), len(resp)
    p_trunc = raw_pl > max_prompt_chars
    r_trunc = raw_rl > max_response_chars
    if p_trunc:
        prompt = prompt[:max_prompt_chars] + "…"
    if r_trunc:
        resp = resp[:max_response_chars] + "…"
    p_note = f"truncated, cap={max_prompt_chars} chars (full {raw_pl})" if p_trunc else f"full, {raw_pl} chars"
    r_note = f"truncated, cap={max_response_chars} chars (full {raw_rl})" if r_trunc else f"full, {raw_rl} chars"
    return (
        f"traj_uid={tu} grpo_index={gi}\n"
        "(prompt/response are tokenizer.decode(skip_special_tokens=True); "
        "standalone lines like system/user/assistant come from the chat template text.)\n"
        f"--- prompt ({p_note}) ---\n{prompt}\n"
        f"--- response ({r_note}) ---\n{resp}\n"
        "--- end mem_adaptor sample ---"
    )


def mem_adaptor_rollout_diag_metrics(
    samples: List[Dict[str, Any]],
    gen_batch_output: DataProto,
) -> Dict[str, float]:
    """Per-rollout diagnostics for adaptor training rows (same ``traj_uid`` / return as GRPO attach).

    Uses ``gen_batch_output.non_tensor_batch`` ``traj_uid``, ``episode_rewards``, and
    ``traj_episode_success`` (if present). Each adaptor sample must carry ``mem_adaptor_reject``
    (single-call ``<EMPTY>``) when available.
    """
    empty = {
        "mean_attached_return": 0.0,
        "reject_rate": 0.0,
        "nonempty_rate": 0.0,
        "success_rate_when_applied": 0.0,
        "success_rate_when_rejected": 0.0,
    }
    if not samples:
        return empty

    tu = gen_batch_output.non_tensor_batch.get("traj_uid")
    if tu is None or len(tu) == 0:
        return empty

    er = gen_batch_output.non_tensor_batch.get("episode_rewards")
    ts = gen_batch_output.non_tensor_batch.get("traj_episode_success")
    uid_map: Dict[str, tuple[float, float]] = {}
    n_rows = len(tu)
    for i in range(n_rows):
        uid = str(tu[i])
        if uid in uid_map:
            continue
        r = (
            float(np.asarray(er[i], dtype=np.float64).reshape(-1)[0])
            if er is not None
            else 0.0
        )
        if ts is not None:
            s = float(np.asarray(ts[i], dtype=np.float64).reshape(-1)[0])
        else:
            s = 1.0 if r > 0.0 else 0.0
        uid_map[uid] = (r, s)

    returns: List[float] = []
    reject_flags: List[float] = []
    succ_when_applied: List[float] = []
    succ_when_rejected: List[float] = []

    for s in samples:
        tid = str(s.get("traj_uid", ""))
        r, succ = uid_map.get(tid, (0.0, 0.0))
        returns.append(r)
        rej = bool(s.get("mem_adaptor_reject", False))
        reject_flags.append(1.0 if rej else 0.0)
        if rej:
            succ_when_rejected.append(succ)
        else:
            succ_when_applied.append(succ)

    n = len(samples)
    return {
        "mean_attached_return": float(np.mean(returns)) if n else 0.0,
        "reject_rate": float(np.mean(reject_flags)) if n else 0.0,
        "nonempty_rate": float(1.0 - np.mean(reject_flags)) if n else 0.0,
        "success_rate_when_applied": float(np.mean(succ_when_applied)) if succ_when_applied else 0.0,
        "success_rate_when_rejected": float(np.mean(succ_when_rejected)) if succ_when_rejected else 0.0,
    }


def train_memory_adaptor_enabled(config: DictConfig) -> bool:
    """True when adaptor GRPO updates should run: enabled, dedicated WG, and ``train_memory_adaptor``."""
    ma = OmegaConf.select(config, "mem_adaptor")
    if ma is None or not bool(ma.get("enable", False)):
        return False
    if bool(ma.get("use_actor_rollout_wg", True)):
        return False
    return bool(ma.get("train_memory_adaptor", False))


def mem_adaptor_kl_loss_enabled(config: DictConfig) -> bool:
    """True when dedicated adaptor GRPO should use KL loss against a frozen ref worker."""
    if not train_memory_adaptor_enabled(config):
        return False
    ma = OmegaConf.select(config, "mem_adaptor")
    if ma is None:
        return False
    if ma.get("actor_use_kl_loss") is not None:
        return bool(ma.get("actor_use_kl_loss"))
    amref = OmegaConf.select(ma, "actor_rollout_ref")
    if amref is None:
        return False
    return bool(OmegaConf.select(amref, "actor.use_kl_loss", default=False))


def mem_adaptor_ref_actor_rollout_config(config: DictConfig) -> DictConfig:
    """Config for colocated adaptor ``role=ref`` worker (frozen KL anchor).

    ``mem_adaptor.ref_model.path`` overrides ``model.path`` when set (e.g. base instruct ckpt).
    """
    amref = config.mem_adaptor.actor_rollout_ref
    ref_cfg = OmegaConf.create(OmegaConf.to_container(amref, resolve=True))
    ref_path = OmegaConf.select(config.mem_adaptor, "ref_model.path", default=None)
    if ref_path is not None and str(ref_path).strip():
        with open_dict(ref_cfg):
            with open_dict(ref_cfg.model):
                ref_cfg.model.path = str(ref_path).strip()
    return ref_cfg


def _traj_uid_to_episode_reward(gen_batch: DataProto) -> Dict[str, float]:
    """Map trajectory uid -> scalar episode return from a post-``gather_rollout_data`` batch."""
    tu = gen_batch.non_tensor_batch.get("traj_uid")
    if tu is None or len(tu) == 0:
        return {}
    n = len(tu)
    er = gen_batch.non_tensor_batch.get("episode_rewards")
    if er is None:
        return {str(tu[i]): 0.0 for i in range(n)}
    out: Dict[str, float] = {}
    for i in range(n):
        v = float(np.asarray(er[i], dtype=np.float64).reshape(-1)[0])
        out[str(tu[i])] = v
    return out


def build_memory_adaptor_grpo_batch(
    *,
    config: DictConfig,
    samples: List[Dict[str, Any]],
    gen_batch_output: DataProto,
    tokenizer: Optional["PreTrainedTokenizer"] = None,
) -> Optional[DataProto]:
    """
    Collate adaptor rollout rows and attach outcome reward on the last non-pad response token.

    Each ``samples`` entry: CPU tensors prompts, responses, input_ids, attention_mask, position_ids;
    int pad_token_id; str traj_uid; ``grpo_index`` = Reasoning GRPO group uid (shared across the
    ``env.rollout.n`` parallel envs) so ``compute_grpo_outcome_advantage`` dedupes one return per
    ``(uid, traj_uid)`` and compares across trajectories in the same group.

    Pass ``tokenizer`` = adaptor tokenizer (same as ``TrajectoryCollector.adaptor_tokenizer``) so
    GRPO auxiliary shaping (e.g. ``grpo_english_shaping``) decodes responses correctly.
    """
    if not samples:
        return None
    rmap = _traj_uid_to_episode_reward(gen_batch_output)
    rl = int(config.mem_adaptor.actor_rollout_ref.rollout.response_length)
    ma = config.mem_adaptor
    rows: List[dict] = []
    n_shaping_pen = 0
    n_shaping_tot = 0
    for s in samples:
        tid = str(s["traj_uid"])
        r = float(rmap.get(tid, 0.0))
        resp = s["responses"]
        if resp.dim() != 1:
            raise ValueError("mem_adaptor training expects 1D responses per row")
        pad_id = int(s["pad_token_id"])
        scores = torch.zeros(rl, dtype=torch.float32)
        # Align to fixed rollout response length (padded in generate_sequences).
        eff = min(int(resp.shape[0]), rl)
        seg = resp[:eff]
        nz = (seg != pad_id).nonzero(as_tuple=True)[0]
        last = int(nz[-1].item()) if nz.numel() > 0 else max(eff - 1, 0)
        last = min(last, rl - 1)
        n_shaping_tot += 1
        dec = _decode_adaptor_response_row(tokenizer, resp, pad_id) if tokenizer is not None else ""
        delta, shaped = _mem_adaptor_grpo_english_shaping_delta(ma, dec)
        if shaped:
            n_shaping_pen += 1
        scores[last] = r + delta
        row = {
            "prompts": s["prompts"],
            "responses": s["responses"],
            "input_ids": s["input_ids"],
            "attention_mask": s["attention_mask"],
            "position_ids": s["position_ids"],
            "token_level_scores": scores,
            "uid": s["grpo_index"],
            "traj_uid": tid,
        }
        rows.append(row)

    data = collate_fn(rows)
    ref_rollout = config.mem_adaptor.actor_rollout_ref.rollout
    meta: Dict[str, Any] = {"temperature": float(ref_rollout.temperature)}
    if n_shaping_tot > 0:
        meta["mem_adaptor_english_shaping_penalized"] = float(n_shaping_pen)
        meta["mem_adaptor_english_shaping_total"] = float(n_shaping_tot)
    return DataProto.from_single_dict(
        data=data,
        meta_info=meta,
    )


def prepare_adaptor_batch_for_rl(batch: DataProto, config: DictConfig) -> DataProto:
    """Attach meta_info for ``compute_log_prob`` / ``update_actor`` on the adaptor worker."""
    ref_rollout = config.mem_adaptor.actor_rollout_ref.rollout
    batch.meta_info["micro_batch_size"] = ref_rollout.log_prob_micro_batch_size_per_gpu
    batch.meta_info["max_token_len"] = ref_rollout.log_prob_max_token_len_per_gpu
    batch.meta_info["use_dynamic_bsz"] = ref_rollout.log_prob_use_dynamic_bsz
    batch.meta_info["temperature"] = float(ref_rollout.temperature)
    batch.meta_info["multi_turn"] = False
    return batch


def pad_mem_adaptor_batch_for_dp(batch: DataProto, world_size: int) -> tuple[DataProto, int]:
    """
    Pad adaptor batches so ``len(batch)`` is divisible by DP ``world_size`` (avoids DataProto.chunk assert).
    Refreshes ``meta_info["global_token_num"]`` when padding adds rows. Returns ``(padded_or_same, pad_size)``.
    """
    ws = int(world_size)
    if ws <= 1 or len(batch) == 0 or len(batch) % ws == 0:
        return batch, 0
    padded, pad_size = pad_dataproto_to_divisor(batch, ws)
    if pad_size and padded.batch is not None and "attention_mask" in padded.batch:
        padded.meta_info["global_token_num"] = torch.sum(padded.batch["attention_mask"], dim=-1).tolist()
    return padded, pad_size
