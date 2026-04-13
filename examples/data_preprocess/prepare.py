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
import argparse
import importlib.util
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='visual', choices=['visual', 'text'])
    parser.add_argument('--local_dir', default='data/MemAdaptor/verl-agent/')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_data_size', default=256, type=int)
    parser.add_argument('--val_data_size', default=256, type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--infer_alfworld_sizes',
        action='store_true',
        help='Set train/val placeholder counts from AlfredTWEnv (full train & eval splits after filtering). '
        'Requires ALFWORLD_DATA / json layout; may take ~1min.',
    )
    parser.add_argument(
        '--alfworld_config',
        default=None,
        help='Path to config_tw.yaml (default: agent_system/.../config_tw.yaml).',
    )
    parser.add_argument(
        '--alfworld_eval_split',
        default='eval_in_distribution',
        choices=['eval_in_distribution', 'eval_out_of_distribution'],
        help='Which eval split to count for val_data_size when using --infer_alfworld_sizes.',
    )

    args = parser.parse_args()

    if args.infer_alfworld_sizes:
        if args.mode != 'text':
            raise ValueError('--infer_alfworld_sizes only supports --mode text')
        _alf_mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alfworld_sizes.py')
        _spec = importlib.util.spec_from_file_location('alfworld_sizes', _alf_mod_path)
        _alf_mod = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_alf_mod)

        n_train, n_val = _alf_mod.infer_alfworld_num_games(args.alfworld_config, args.alfworld_eval_split)
        args.train_data_size = n_train
        args.val_data_size = n_val
        print(
            f"[prepare] infer_alfworld_sizes: train_games={args.train_data_size}, "
            f"eval_games({args.alfworld_eval_split})={args.val_data_size}"
        )

    print(f"processing data for mode: {args.mode}")
    args.local_dir = os.path.join(os.path.expanduser(args.local_dir), args.mode)
    os.makedirs(args.local_dir, exist_ok=True)

    train_output_path = os.path.join(args.local_dir, 'train.parquet')
    test_output_path = os.path.join(args.local_dir, 'test.parquet')

    if not args.overwrite and os.path.exists(train_output_path) and os.path.exists(test_output_path):
        print(f"found existing parquet files under {args.local_dir}, reusing them")
        print(
            f"[prepare] train_samples={args.train_data_size}, val_samples={args.val_data_size}, "
            f"total_samples={args.train_data_size + args.val_data_size} "
            f"(requested sizes; parquet reused, use --overwrite to regenerate)"
        )
        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            copy(src=args.local_dir, dst=args.hdfs_dir)
        raise SystemExit(0)

    data_source = 'hiyouga/geometry3k'
    """
    **NOTE**: This is a frequently asked question.
    We do NOT use the data in 'hiyouga/geometry3k', instead we only use it to indicate the modality and the data size.
    See details: https://github.com/langfengQ/verl-agent?tab=readme-ov-file#2-data-preparation
    """

    if args.mode == 'text':
        print("text mode uses synthetic placeholder samples, skipping remote dataset download")
        train_dataset = datasets.Dataset.from_dict({'placeholder': [''] * args.train_data_size})
        test_dataset = datasets.Dataset.from_dict({'placeholder': [''] * args.val_data_size})
    else:
        dataset = datasets.load_dataset(data_source)
        train_dataset = dataset['train'].select(range(args.train_data_size))
        test_dataset = dataset['test'].select(range(args.val_data_size))

    instruction_following = {
        "visual": "<image>",
        "text": "",
        }

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = instruction_following[args.mode]

            if args.mode == 'visual':
                if 'images' not in example:
                    raise KeyError(f"'images' field not found in visual dataset example: {list(example.keys())}")
                data = {
                    "data_source": args.mode,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "images": example['images'],
                    "ability": "agent",
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            else:
                data = {
                    "data_source": args.mode,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "ability": "agent",
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)

    print(
        f"[prepare] train_samples={len(train_dataset)}, val_samples={len(test_dataset)}, "
        f"total_samples={len(train_dataset) + len(test_dataset)}"
    )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(train_output_path)
    test_dataset.to_parquet(test_output_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
