#!/usr/bin/env python3
"""
Randomly sample n rows from a cold-start parquet (e.g. 20260429.parquet) and write
both .parquet and .jsonl with the same columns.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd


def convert_to_jsonable(v: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(v, np.ndarray):
            # ndarray.item() only works for size-1 arrays; use nested Python types instead.
            return convert_to_jsonable(v.tolist())
        if isinstance(v, np.generic):
            return v.item()

    if isinstance(v, dict):
        return {str(k): convert_to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [convert_to_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    # e.g. torch scalar Tensor
    if hasattr(v, "item") and callable(getattr(v, "item")):
        try:
            return v.item()
        except Exception:
            pass
    return str(v)


def write_jsonl(path: str, frame: pd.DataFrame) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as jf:
        for _, row in frame.iterrows():
            obj = {k: convert_to_jsonable(row[k]) for k in frame.columns}
            jf.write(json.dumps(obj, ensure_ascii=False) + "\n")


def default_prefix(input_path: str, n_take: int, seed: int) -> str:
    base = os.path.splitext(os.path.abspath(input_path))[0]
    return f"{base}_sample{n_take}_seed{seed}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/MemAdaptor/cold_start/webshop/20260511.parquet",
        help="Source parquet path.",
    )
    p.add_argument("--n", type=int, default=1100, help="Number of rows to sample (capped by dataset size).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    p.add_argument(
        "--out-prefix",
        default="",
        help=(
            "Output path without extension; writes <out-prefix>.parquet and <out-prefix>.jsonl. "
            "Default: <input_stem>_sample<n>_seed<seed>. "
            "If --val-n > 0, writes <prefix>_train.* and <prefix>_val.* instead."
        ),
    )
    p.add_argument(
        "--val-n",
        type=int,
        default=100,
        help=(
            "If >0, reserve this many rows as validation after sampling (disjoint from train). "
            "Requires n > val-n. Outputs *_train.parquet/jsonl and *_val.parquet/jsonl."
        ),
    )
    args = p.parse_args()

    inp = args.input.strip()
    if not inp:
        raise SystemExit("--input is empty")
    if args.n < 1:
        raise SystemExit("--n must be >= 1")

    df = pd.read_parquet(inp)
    if len(df) == 0:
        raise SystemExit("Input parquet has no rows")

    n_take = min(args.n, len(df))
    if n_take < args.n:
        print(f"Warning: only {len(df)} rows available; sampling {n_take} instead of {args.n}.", flush=True)

    base = args.out_prefix.strip() or default_prefix(inp, n_take, args.seed)

    sampled = df.sample(n=n_take, random_state=args.seed).reset_index(drop=True)

    def write_pair(prefix: str, frame: pd.DataFrame) -> None:
        pq = f"{prefix}.parquet"
        jl = f"{prefix}.jsonl"
        parent = os.path.dirname(pq)
        if parent:
            os.makedirs(parent, exist_ok=True)
        frame.to_parquet(pq, index=False)
        write_jsonl(jl, frame)
        print(f"Wrote {len(frame)} rows -> {pq}")
        print(f"Wrote {len(frame)} rows -> {jl}")

    val_n = int(args.val_n)
    if val_n > 0:
        if val_n >= len(sampled):
            raise SystemExit(f"--val-n ({val_n}) must be < sampled rows ({len(sampled)})")
        train_df = sampled.iloc[: len(sampled) - val_n].reset_index(drop=True)
        val_df = sampled.iloc[len(sampled) - val_n :].reset_index(drop=True)
        write_pair(f"{base}_train", train_df)
        write_pair(f"{base}_val", val_df)
    else:
        write_pair(base, sampled)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
