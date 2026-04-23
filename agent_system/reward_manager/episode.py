# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from verl import DataProto

from agent_system.reward_manager.format_reward import (
    FORMAT_REWARD_EXTRA_KEYS,
    compute_generic_action_think_memory_format_reward_multi_step,
    compute_search_think_memory_format_reward_multi_step,
    empty_format_reward_metrics,
    use_search_format_reward,
)

# EpisodeRewardManager stdout sampling defaults (kept in code; not duplicated in ppo_trainer.yaml).
DEFAULT_NUM_EXAMINE_TRAIN = 20
DEFAULT_NUM_EXAMINE_VAL = 0
DEFAULT_EPISODE_TRAJ_SAMPLE_EVERY_N_TRAINER_STEPS = 3


def _mem_query_flag(x) -> bool:
    if x is None:
        return False
    try:
        return bool(np.asarray(x, dtype=bool).reshape(-1)[0])
    except Exception:
        return bool(x)


def _empty_retrieval_message_strip(cfg: DictConfig) -> str:
    try:
        raw = OmegaConf.select(cfg, "env.memory.empty_retrieval_message", default="")
        return str(raw or "").strip()
    except Exception:
        return ""


def _traj_memory_flags(rows: list[dict], empty_inject_msg: str) -> tuple[bool, bool]:
    """(has_retrieval, has_nonempty_recall) for one trajectory's rows."""
    mrc0 = rows[0].get("memory_retrieval_counts")
    try:
        mrc = float(np.asarray(mrc0, dtype=np.float64).reshape(-1)[0])
    except Exception:
        mrc = 0.0
    has_retrieval = mrc > 0.0
    if not has_retrieval:
        for r in rows:
            if _mem_query_flag(r.get("memory_query_mask")):
                has_retrieval = True
                break
    has_hit = False
    for r in rows:
        inj = r.get("memory_injected_text")
        if inj is None:
            continue
        s = str(inj).strip()
        if s and s != empty_inject_msg:
            has_hit = True
            break
    return has_retrieval, has_hit


class EpisodeRewardManager:
    """Episode return from the env plus optional format reward (see ``reward_model.format_reward``)."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        normalize_by_length: bool,
        config: DictConfig,
    ) -> None:
        if not isinstance(config, DictConfig):
            raise TypeError("EpisodeRewardManager requires config: DictConfig")
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length
        self.config = config
        self._fr = config.reward_model.format_reward

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)

        fr = self._fr
        fr_enabled = bool(fr.enable)
        w_outcome = float(fr.weight_outcome)
        w_format = float(fr.weight_format)
        search_markers: list[str] = list(
            OmegaConf.to_container(fr.search_data_source_substrings, resolve=True)
        )

        row_infos: list[dict] = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            api_txt = data_item.non_tensor_batch.get("api_response_text")
            if api_txt is not None:
                response_str = str(api_txt)
            else:
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch["data_source"]

            episode_rewards = data_item.non_tensor_batch["episode_rewards"]
            episode_lengths = data_item.non_tensor_batch["episode_lengths"]

            if self.normalize_by_length:
                raw_outcome = episode_rewards / episode_lengths
            else:
                raw_outcome = episode_rewards
            outcome = float(np.asarray(raw_outcome, dtype=np.float64).reshape(-1)[0])

            tu = data_item.non_tensor_batch.get("traj_uid")
            traj_uid_s = str(tu) if tu is not None else str(i)
            ntb = data_item.non_tensor_batch

            row_infos.append(
                {
                    "i": i,
                    "traj_uid": traj_uid_s,
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "data_source": data_source,
                    "outcome": outcome,
                    "valid_response_length": int(valid_response_length),
                    "prompt_ids": prompt_ids,
                    "memory_query_mask": ntb.get("memory_query_mask"),
                    "memory_injected_text": ntb.get("memory_injected_text"),
                    "memory_retrieval_counts": ntb.get("memory_retrieval_counts"),
                }
            )

        by_traj: dict[str, list[dict]] = defaultdict(list)
        for r in row_infos:
            by_traj[r["traj_uid"]].append(r)
        for rows in by_traj.values():
            rows.sort(key=lambda x: x["i"])

        meta = data.meta_info or {}
        trainer_step_raw = meta.get("trainer_global_step")
        try:
            trainer_step_i = int(trainer_step_raw) if trainer_step_raw is not None else None
        except (TypeError, ValueError):
            trainer_step_i = None
        warmup_steps = int(OmegaConf.select(fr, "format_warmup_global_steps", default=0) or 0)
        in_format_warmup = (
            warmup_steps > 0
            and trainer_step_i is not None
            and trainer_step_i < warmup_steps
        )
        warmup_wfmt_mult = float(OmegaConf.select(fr, "warmup_weight_format_multiplier", default=0.0) or 0.0)
        warmup_req_mem = OmegaConf.select(fr, "warmup_require_memory_retrieve", default=None)
        warmup_pen_cn = OmegaConf.select(fr, "warmup_penalize_chinese_chars", default=None)

        traj_format: dict[str, tuple[float, dict[str, float]]] = {}
        for tu, rows in by_traj.items():
            format_score = 0.0
            fmt_metrics = empty_format_reward_metrics()
            applied_format = False
            w_format_step = w_format
            if fr_enabled and w_format != 0.0:
                applied_format = True
                if in_format_warmup:
                    w_format_step = w_format * warmup_wfmt_mult
                min_seg = int(fr.min_segment_chars)
                req_mem = bool(fr.require_memory_retrieve)
                if in_format_warmup and warmup_req_mem is not None:
                    req_mem = bool(warmup_req_mem)
                w_t = float(fr.weight_think)
                w_a = float(fr.weight_action)
                w_m = float(fr.weight_memory)
                pen_cn = bool(fr.penalize_chinese_chars)
                if in_format_warmup and warmup_pen_cn is not None:
                    pen_cn = bool(warmup_pen_cn)
                max_think = int(fr.max_think_segments)
                max_action = int(fr.max_action_segments)
                max_mem_seg = int(fr.max_memory_retrieve_segments)
                data_source = rows[0]["data_source"]
                step_texts = [str(x["response_str"]) for x in rows]
                if use_search_format_reward(str(data_source), search_markers):
                    out = compute_search_think_memory_format_reward_multi_step(
                        step_texts,
                        min_segment_chars=min_seg,
                        require_memory_retrieve=req_mem,
                        memory_open_tag=str(fr.memory_open_tag),
                        memory_close_tag=str(fr.memory_close_tag),
                        think_open_tag=str(fr.think_open_tag),
                        think_close_tag=str(fr.think_close_tag),
                        search_open_tag=str(fr.search_open_tag),
                        search_close_tag=str(fr.search_close_tag),
                        answer_open_tag=str(fr.answer_open_tag),
                        answer_close_tag=str(fr.answer_close_tag),
                        weight_think=w_t,
                        weight_protocol=w_a,
                        weight_memory=w_m,
                        penalize_chinese_chars=pen_cn,
                        max_think_segments=max_think,
                        max_memory_retrieve_segments=max_mem_seg,
                    )
                else:
                    out = compute_generic_action_think_memory_format_reward_multi_step(
                        step_texts,
                        min_segment_chars=min_seg,
                        require_memory_retrieve=req_mem,
                        memory_open_tag=str(fr.memory_open_tag),
                        memory_close_tag=str(fr.memory_close_tag),
                        think_open_tag=str(fr.think_open_tag),
                        think_close_tag=str(fr.think_close_tag),
                        action_open_tag=str(fr.action_open_tag),
                        action_close_tag=str(fr.action_close_tag),
                        weight_think=w_t,
                        weight_action=w_a,
                        weight_memory=w_m,
                        penalize_chinese_chars=pen_cn,
                        max_think_segments=max_think,
                        max_action_segments=max_action,
                        max_memory_retrieve_segments=max_mem_seg,
                    )
                format_score = float(out.reward)
                fmt_metrics = out.metrics
            traj_format[tu] = (format_score if applied_format else 0.0, fmt_metrics, applied_format)

        # Print one full trajectory per qualifying reward batch (num_examine > 0), throttled by global step.
        _every = max(1, int(DEFAULT_EPISODE_TRAJ_SAMPLE_EVERY_N_TRAINER_STEPS))
        _do_traj_print = self.num_examine > 0 and bool(by_traj)
        if _do_traj_print and _every > 1:
            if trainer_step_i is None:
                _do_traj_print = False
            elif trainer_step_i % _every != 0:
                _do_traj_print = False
        if _do_traj_print:
            empty_inj = _empty_retrieval_message_strip(self.config)
            all_uids = list(by_traj.keys())
            tier_hit: list[str] = []
            tier_retrieve_only: list[str] = []
            for tu_k in all_uids:
                hr, hh = _traj_memory_flags(by_traj[tu_k], empty_inj)
                if hr and hh:
                    tier_hit.append(tu_k)
                elif hr:
                    tier_retrieve_only.append(tu_k)
            pick_pool = tier_hit or tier_retrieve_only or all_uids
            tier_name = "retrieve+recall" if tier_hit else ("retrieve_only" if tier_retrieve_only else "fallback")
            tu_pick = random.choice(pick_pool)
            ds_pick = by_traj[tu_pick][0]["data_source"]
            fs_pick, _, app_pick = traj_format[tu_pick]
            out_pick = float(by_traj[tu_pick][0]["outcome"])
            wfs = w_format
            if fr_enabled and w_format != 0.0:
                if in_format_warmup:
                    wfs = w_format * warmup_wfmt_mult
            fin_pick = (
                w_outcome * out_pick + wfs * fs_pick
                if fr_enabled and app_pick
                else w_outcome * out_pick
            )
            print(
                f"[{ds_pick}][traj_sample] uid={tu_pick} tier={tier_name} "
                f"outcome={out_pick} format={fs_pick} final={fin_pick}",
                flush=True,
            )
            for r in by_traj[tu_pick]:
                print(f"[{ds_pick}][prompt]", r["prompt_str"], flush=True)
                print(f"[{ds_pick}][response]", r["response_str"], flush=True)
                print(
                    f"[{ds_pick}][outcome]",
                    r["outcome"],
                    "[format]",
                    fs_pick,
                    "[final]",
                    fin_pick,
                    flush=True,
                )

        for r in row_infos:
            i = r["i"]
            tu = r["traj_uid"]
            prompt_str = r["prompt_str"]
            response_str = r["response_str"]
            data_source = r["data_source"]
            outcome = r["outcome"]
            valid_response_length = r["valid_response_length"]
            prompt_ids = r["prompt_ids"]

            format_score, fmt_metrics, applied_format = traj_format[tu]

            if fr_enabled and applied_format:
                final_score = w_outcome * outcome + w_format_step * format_score
            else:
                final_score = w_outcome * outcome

            reward_tensor[i, valid_response_length - 1] = torch.tensor(
                final_score, dtype=torch.float32, device=prompt_ids.device
            )

            reward_extra_info["outcome_reward_scalar"].append(outcome)
            reward_extra_info["format_reward_scalar"].append(format_score if applied_format else 0.0)
            reward_extra_info["format_reward_applied"].append(1.0 if applied_format else 0.0)
            reward_extra_info["combined_reward_scalar"].append(final_score)
            for k in FORMAT_REWARD_EXTRA_KEYS:
                reward_extra_info[k].append(float(fmt_metrics[k]))

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
