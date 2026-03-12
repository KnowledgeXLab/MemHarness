# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.agentgym.client import init_env_client
logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        agentgym_config: DictConfig,
    ):

        self.data_file = copy.deepcopy(data_file)
        self.original_data_file = copy.deepcopy(data_file)  # use for resume
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.agentgym_config = agentgym_config

        self.cache_dir = os.path.expanduser(data_config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = data_config.get("prompt_key", "prompt")
        self.image_key = data_config.get("image_key", "images")
        self.video_key = data_config.get("video_key", "videos")
        self.max_prompt_length = data_config.get("max_prompt_length", 1024)

        self.return_raw_chat = data_config.get("return_raw_chat", False)
        self.return_full_prompt = data_config.get("return_full_prompt", False)
        self.truncation = data_config.get("truncation", "error")
        self.filter_overlong_prompts = data_config.get("filter_overlong_prompts", True)

        self.num_workers = data_config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = data_config.get("chat_template_func", None)
        self.need_tools_kwargs = data_config.get("need_tools_kwargs", False)
        self.filter_prompts = data_config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._read_files_and_tokenize()
        # get agentgym client
        self.env_client = init_env_client(self.agentgym_config)

    def _read_files_and_tokenize(self):
        self.dataframe = datasets.load_dataset("json", data_files=self.data_file)["train"]
        print(f"dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_file")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        example["data_source"] = example[self.prompt_key].split("_")[0]
        messages = [{"role": "user", "content": self.env_client.conversation_start[0]["value"]},
                     {"role": "assistant", "content": self.env_client.conversation_start[1]["value"]}]
        prompt_with_chat_template = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + self.env_client.conversation_start[0]["value"] + "<|im_end|>\n<|im_start|>assistant\n" + self.env_client.conversation_start[1]["value"] + "<|im_end|>"
        return messages, prompt_with_chat_template

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        
        messages, prompt_with_chat_template = self._build_messages(row_dict)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
