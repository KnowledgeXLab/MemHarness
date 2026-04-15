# Copyright 2025 MemAdaptor
#
# Train the Memory Adaptor policy with trajectory-level rewards (GRPO).
# Requires ``use_actor_rollout_wg=false`` (dedicated adaptor WorkerGroup). Reasoning may be frozen
# (``actor_rollout_ref.actor.trainable=false``) or trained in the same step (joint: leave Reasoning trainable).

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from collections import Counter

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.dataset.rl_dataset import collate_fn


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
    max_prompt_chars: int = 800,
    max_response_chars: int = 800,
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
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars] + "…"
    if len(resp) > max_response_chars:
        resp = resp[:max_response_chars] + "…"
    return (
        f"traj_uid={tu} grpo_index={gi}\n"
        f"--- prompt (truncated to {max_prompt_chars} chars) ---\n{prompt}\n"
        f"--- response (truncated to {max_response_chars} chars) ---\n{resp}"
    )


def train_memory_adaptor_enabled(config: DictConfig) -> bool:
    """True when adaptor GRPO updates should run: enabled, dedicated WG, and ``train_memory_adaptor``."""
    ma = OmegaConf.select(config, "mem_adaptor")
    if ma is None or not bool(ma.get("enable", False)):
        return False
    if bool(ma.get("use_actor_rollout_wg", True)):
        return False
    return bool(ma.get("train_memory_adaptor", False))


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
) -> Optional[DataProto]:
    """
    Collate adaptor rollout rows and attach outcome reward on the last non-pad response token.

    Each ``samples`` entry: CPU tensors prompts, responses, input_ids, attention_mask, position_ids;
    int pad_token_id; str traj_uid; ``grpo_index`` = Reasoning GRPO group uid (shared across the
    ``env.rollout.n`` parallel envs) so ``compute_grpo_outcome_advantage`` dedupes one return per
    ``(uid, traj_uid)`` and compares across trajectories in the same group.
    """
    if not samples:
        return None
    rmap = _traj_uid_to_episode_reward(gen_batch_output)
    rl = int(config.mem_adaptor.actor_rollout_ref.rollout.response_length)
    rows: List[dict] = []
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
        scores[last] = r
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
    return DataProto.from_single_dict(
        data=data,
        meta_info={"temperature": float(ref_rollout.temperature)},
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
