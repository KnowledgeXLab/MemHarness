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


def _hdfs_upload(local_dir: str, hdfs_dir: str | None) -> None:
    if hdfs_dir is None:
        return
    from verl.utils.hdfs_io import copy, makedirs

    makedirs(hdfs_dir)
    copy(src=local_dir, dst=hdfs_dir)


def _write_text_agent_parquet(train_path: str, test_path: str, n_train: int, n_val: int) -> None:
    """Write RL agent placeholder parquet without importing HuggingFace ``datasets``.

    Some conda/env combinations segfault during ``import datasets`` (pyarrow/cpp).
    Text placeholders only need a minimal parquet schema compatible with verl RL loaders.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "text-mode prepare requires pyarrow. Install e.g. `pip install pyarrow` or `conda install pyarrow`."
        ) from e

    rows_train = [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": "train", "index": idx},
        }
        for idx in range(n_train)
    ]
    rows_test = [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": "test", "index": idx},
        }
        for idx in range(n_val)
    ]
    pq.write_table(pa.Table.from_pylist(rows_train), train_path)
    pq.write_table(pa.Table.from_pylist(rows_test), test_path)


def _webshop_json_paths_for_infer(args):
    """Resolve items_shuffle / items_ins JSON paths from ``--webshop-data-dir`` (if set)."""
    data_dir = args.webshop_data_dir
    if not data_dir:
        return None, None
    data_dir = os.path.expanduser(str(data_dir))
    if args.webshop_use_small:
        return (
            os.path.join(data_dir, 'items_shuffle_1000.json'),
            os.path.join(data_dir, 'items_ins_v2_1000.json'),
        )
    return (
        os.path.join(data_dir, 'items_shuffle.json'),
        os.path.join(data_dir, 'items_ins_v2.json'),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='visual', choices=['visual', 'text'])
    parser.add_argument(
        '--local_dir',
        default='data/MemHarness/verl-agent/',
        help='Output root; files go to <local_dir>/<mode>/ (use per-bench roots e.g. data/verl-agent/alfworld).',
    )
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
    parser.add_argument(
        '--infer_webshop_sizes',
        action='store_true',
        help='Set train/val placeholder counts from WebShop goal splits (train: goals 500..N-1, '
        'val: first 500 indices), matching WebshopMultiProcessEnv. Uses lightweight JSON counting '
        '(no Gym).',
    )
    parser.add_argument(
        '--webshop-use-small',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='When inferring WebShop sizes: expect items_shuffle_1000 / items_ins_v2_1000 under '
        '--webshop-data-dir (default true, matches env.webshop.use_small).',
    )
    parser.add_argument(
        '--webshop-human-goals',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Must match Hydra env.webshop.human_goals. False (default) = synthetic goals '
        '(matches ppo_trainer.yaml). True = human goal instructions from items_human_ins.json.',
    )
    parser.add_argument(
        '--webshop-data-dir',
        default=os.environ.get('WEBSHOP_DATA_DIR') or os.environ.get('MEMHARNESS_WEBSHOP_DATA_DIR'),
        metavar='DIR',
        help='Directory containing WebShop items_shuffle*.json and items_ins_v2*.json (same names as '
        'vendored webshop/data/). Used by --infer_webshop_sizes; if unset, uses paths under the repo. '
        'Environment: WEBSHOP_DATA_DIR or MEMHARNESS_WEBSHOP_DATA_DIR.',
    )

    args = parser.parse_args()

    if args.infer_alfworld_sizes and args.infer_webshop_sizes:
        raise ValueError('Use only one of --infer_alfworld_sizes or --infer_webshop_sizes')

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

    if args.infer_webshop_sizes:
        if args.mode != 'text':
            raise ValueError('--infer_webshop_sizes only supports --mode text')
        _ws_mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webshop_sizes.py')
        _spec = importlib.util.spec_from_file_location('webshop_sizes', _ws_mod_path)
        _ws_mod = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_ws_mod)

        ws_fp, ws_ap = _webshop_json_paths_for_infer(args)
        n_train, n_val = _ws_mod.infer_webshop_num_tasks(
            use_small=args.webshop_use_small,
            human_goals=args.webshop_human_goals,
            file_path=ws_fp,
            attr_path=ws_ap,
        )
        args.train_data_size = n_train
        args.val_data_size = n_val
        print(
            f"[prepare] infer_webshop_sizes: train_goals(>=500 index)={args.train_data_size}, "
            f"val_goals(first 500 indices)={args.val_data_size}"
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
        _hdfs_upload(args.local_dir, args.hdfs_dir)
        raise SystemExit(0)

    data_source = 'hiyouga/geometry3k'
    """
    **NOTE**: This is a frequently asked question.
    We do NOT use the data in 'hiyouga/geometry3k', instead we only use it to indicate the modality and the data size.
    See details: https://github.com/langfengQ/verl-agent?tab=readme-ov-file#2-data-preparation
    """

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if args.mode == 'text':
        print("text mode uses synthetic placeholder samples, skipping remote dataset download")
        _write_text_agent_parquet(train_output_path, test_output_path, args.train_data_size, args.val_data_size)
        print(
            f"[prepare] train_samples={args.train_data_size}, val_samples={args.val_data_size}, "
            f"total_samples={args.train_data_size + args.val_data_size}"
        )
    else:
        import datasets

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

        train_dataset.to_parquet(train_output_path)
        test_dataset.to_parquet(test_output_path)

    _hdfs_upload(local_dir, hdfs_dir)
