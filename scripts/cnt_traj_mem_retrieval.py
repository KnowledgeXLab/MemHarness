#!/usr/bin/env python3
"""
统计 ``_dump_validation_trajectories_jsonl``（validation_data_dir）保存的 JSONL：
每行一条轨迹，根据 ``memory_retrieval_count > 0`` 统计「发生过记忆检索」的条数。

对应 ``verl/trainer/ppo/ray_trainer.py`` 中的 dump 格式。

用法::

    python scripts/count_memory_retrieval_in_trajectories.py /path/to/file.jsonl
    python scripts/count_memory_retrieval_in_trajectories.py /path/to/dir --glob "*.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Iterable


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def _float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def count_file(path: str) -> tuple[int, int]:
    """每行一条验证轨迹。返回 (总条数, memory_retrieval_count > 0 的条数)。"""
    total = 0
    with_retrieval = 0
    for obj in _iter_jsonl(path):
        total += 1
        c = _float_or_none(obj.get("memory_retrieval_count"))
        if c is not None and c > 0:
            with_retrieval += 1
    return total, with_retrieval


def main() -> None:
    ap = argparse.ArgumentParser(
        description="统计 validation JSONL（_dump_validation_trajectories_jsonl）中带记忆检索的轨迹数"
    )
    ap.add_argument("--path", default="data/exp_results/MemAdaptor/pre_exp/alfworld/Qwen2.5-1.5B-Instruct-with_agentic_memory-retrieve_memory_text/val_traj/0.jsonl", type=str, help="jsonl 文件路径，或包含 jsonl 的目录")
    ap.add_argument(
        "--glob",
        dest="glob_pat",
        default="*.jsonl",
        help="path 为目录时的匹配模式（默认 *.jsonl）",
    )
    args = ap.parse_args()

    if os.path.isdir(args.path):
        pattern = os.path.join(args.path, args.glob_pat)
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"未找到匹配文件: {pattern}")
            return
        for fp in files:
            total, with_r = count_file(fp)
            pct = (100.0 * with_r / total) if total else 0.0
            print(f"{fp}")
            print(f"  总轨迹数: {total}")
            print(f"  memory_retrieval_count > 0: {with_r} ({pct:.2f}%)")
            print()
    else:
        total, with_r = count_file(args.path)
        pct = (100.0 * with_r / total) if total else 0.0
        print(f"总轨迹数: {total}")
        print(f"memory_retrieval_count > 0: {with_r} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
