# Copyright 2025 MemAdaptor
#
# Train the Memory Adaptor policy with trajectory-level rewards (GRPO).
# Requires ``use_actor_rollout_wg=false`` (dedicated adaptor WorkerGroup). Reasoning may be frozen
# (``actor_rollout_ref.actor.trainable=false``) or trained in the same step (joint: leave Reasoning trainable).

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn


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


def train_memory_adaptor_enabled(config: DictConfig) -> bool:
    ma = OmegaConf.select(config, "mem_adaptor")
    if ma is None or not bool(ma.get("enable", False)):
        return False
    if bool(ma.get("use_actor_rollout_wg", True)):
        return False
    if "train_memory_adaptor" in ma:
        return bool(ma.get("train_memory_adaptor"))
    return bool(ma.get("trainable", False))


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
    int pad_token_id; str traj_uid, grpo_index.
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
