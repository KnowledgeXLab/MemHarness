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

import re
import os
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification

import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', required=True, type = str, help="The path for your saved model")
    parser.add_argument('--save_dir', default=None, type = str, help="The path to save your model")
    parser.add_argument("--hf_upload_path", default=False, type = str, help="The path of the huggingface repo to upload")
    args = parser.parse_args()

    assert not args.local_dir.endswith("huggingface"), "The local_dir should not end with huggingface"
    local_dir = args.local_dir
    save_dir = args.save_dir

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)  
            break  
    assert world_size, "No model file with the proper format"
        
    state_dict = torch.load(os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt'), map_location='cpu')

    print('Writing to local disk')
    if save_dir and os.path.abspath(save_dir) != os.path.abspath(local_dir):
        hf_path = os.path.join(save_dir, 'huggingface')
        shutil.copytree(os.path.join(local_dir, 'huggingface'), hf_path)
    else:
        hf_path = os.path.join(local_dir, 'huggingface')
    config = AutoConfig.from_pretrained(hf_path)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model
    if args.hf_upload_path:
        # Push to hugging face
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
        api.upload_folder(
            folder_path=hf_path,
            repo_id=args.hf_upload_path,
            repo_type="model"
        )
    
    





