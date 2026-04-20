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

from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from verl import DataProto

from agent_system.reward_manager.format_reward import (
    FORMAT_REWARD_EXTRA_KEYS,
    compute_generic_action_think_memory_format_reward,
    compute_search_think_memory_format_reward,
    empty_format_reward_metrics,
    use_search_format_reward,
)


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

        already_print_data_sources = {}
        fr = self._fr
        fr_enabled = bool(fr.enable)
        w_outcome = float(fr.weight_outcome)
        w_format = float(fr.weight_format)
        search_markers: list[str] = list(
            OmegaConf.to_container(fr.search_data_source_substrings, resolve=True)
        )

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
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

            fmt_metrics = empty_format_reward_metrics()
            format_score = 0.0
            applied_format = False

            if fr_enabled and w_format != 0.0:
                applied_format = True
                min_seg = int(fr.min_segment_chars)
                req_mem = bool(fr.require_memory_retrieve)
                w_t = float(fr.weight_think)
                w_a = float(fr.weight_action)
                w_m = float(fr.weight_memory)
                pen_cn = bool(fr.penalize_chinese_chars)
                if use_search_format_reward(str(data_source), search_markers):
                    out = compute_search_think_memory_format_reward(
                        response_str,
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
                    )
                else:
                    out = compute_generic_action_think_memory_format_reward(
                        response_str,
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
                    )
                format_score = float(out.reward)
                fmt_metrics = out.metrics

            if fr_enabled and applied_format:
                final_score = w_outcome * outcome + w_format * format_score
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

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][outcome]", outcome, "[format]", format_score, "[final]", final_score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
