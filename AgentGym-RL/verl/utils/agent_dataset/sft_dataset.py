# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 json_file: str,
                 tokenizer,
                 prompt_key='conversations',
                 max_length=4096,
                 truncation='right'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        self.json_file = json_file
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key

        self.max_length = max_length

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        self.dataframe = pd.read_json(self.json_file)
        self.prompts = self.dataframe[self.prompt_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]

        # string
        system_chat_dict = {'role': 'system', 'content': ''}
        system_chat_str = tokenizer.apply_chat_template([system_chat_dict], tokenize=False)
        prompt_ids_output = tokenizer(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt[0]['value']}], tokenize=False), return_tensors='pt', add_special_tokens=False)
        input_ids = prompt_ids_output['input_ids'][0]
        attention_mask = prompt_ids_output['attention_mask'][0]
        loss_mask = torch.zeros_like(input_ids)
        for c in prompt[1:]:
            if c['from'] == 'system':
                prompt_ids_output = tokenizer(tokenizer.apply_chat_template([system_chat_dict, {'role': 'system', 'content': c['value']}], tokenize=False).replace(system_chat_str, ""), return_tensors='pt', add_special_tokens=False)
                input_ids = torch.concat([input_ids, prompt_ids_output['input_ids'][0]])
                attention_mask = torch.concat([attention_mask, prompt_ids_output['attention_mask'][0]])
                loss_mask = torch.cat([loss_mask, torch.zeros_like(prompt_ids_output['input_ids'][0])])
            elif c['from'] == 'human':
                prompt_ids_output = tokenizer(tokenizer.apply_chat_template([system_chat_dict, {'role': 'user', 'content': c['value']}], tokenize=False).replace(system_chat_str, ""), return_tensors='pt', add_special_tokens=False)
                input_ids = torch.concat([input_ids, prompt_ids_output['input_ids'][0]])
                attention_mask = torch.concat([attention_mask, prompt_ids_output['attention_mask'][0]])
                loss_mask = torch.cat([loss_mask, torch.zeros_like(prompt_ids_output['input_ids'][0])])
            elif c['from'] == 'gpt':
                prompt_ids_output = tokenizer(tokenizer.apply_chat_template([system_chat_dict, {'role': 'assistant', 'content': c['value']}], tokenize=False).replace(system_chat_str, ""), return_tensors='pt', add_special_tokens=False)
                input_ids = torch.concat([input_ids, prompt_ids_output['input_ids'][0]])
                attention_mask = torch.concat([attention_mask, prompt_ids_output['attention_mask'][0]])
                loss_mask = torch.cat([loss_mask, torch.ones_like(prompt_ids_output['input_ids'][0])])
            else:
                raise NotImplementedError

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }
