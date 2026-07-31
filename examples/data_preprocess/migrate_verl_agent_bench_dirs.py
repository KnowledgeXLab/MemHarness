#!/usr/bin/env python3
"""Migrate legacy shared verl-agent/text parquet to per-bench directories.

Legacy layout:  <data>/verl-agent/{bench}/text/ or <data>/MemAdaptor/verl-agent/text/
New layout:      <data>/MemHarness/verl-agent/{alfworld,webshop}/text/{train,test}.parquet

If legacy parquet is readable, copies to the matching bench by train row count.
Skips targets that already have both parquet files. Does not overwrite existing bench data.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def _repo_data_dir(repo_root: str) -> str:
    link = os.path.join(repo_root, "data")
    if os.path.islink(link):
        return os.path.realpath(link)
    os.makedirs(link, exist_ok=True)
    return os.path.realpath(link)


def _parquet_rows(path: str) -> int | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[migrate] pyarrow not installed; cannot inspect legacy parquet", file=sys.stderr)
        return None
    try:
        return pq.read_table(path).num_rows
    except Exception as exc:
        print(f"[migrate] skip unreadable {path}: {exc}")
        return None


def _bench_for_train_rows(n_train: int) -> str | None:
    # Typical placeholder counts from prepare --infer_*_sizes (2026-07).
    if 3000 <= n_train <= 4000:
        return "alfworld"
    if 6000 <= n_train <= 7000:
        return "webshop"
    return None


def _copy_pair(src_text_dir: str, dst_text_dir: str, *, dry_run: bool) -> None:
    os.makedirs(dst_text_dir, exist_ok=True)
    for name in ("train.parquet", "test.parquet"):
        src = os.path.join(src_text_dir, name)
        dst = os.path.join(dst_text_dir, name)
        if dry_run:
            print(f"[migrate] would copy {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"[migrate] copied {src} -> {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("MEMHARNESS_REPO_ROOT")
        or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = _repo_data_dir(args.repo_root)
    mem_root = os.path.join(data_dir, "MemHarness", "verl-agent")

    legacy_sources: list[tuple[str, str | None]] = [
        (os.path.join(data_dir, "MemAdaptor", "verl-agent", "text"), None),
        (os.path.join(data_dir, "verl-agent", "text"), None),
        (os.path.join(data_dir, "verl-agent", "alfworld", "text"), "alfworld"),
        (os.path.join(data_dir, "verl-agent", "webshop", "text"), "webshop"),
    ]

    copied = 0
    for src_text, bench_hint in legacy_sources:
        src_train = os.path.join(src_text, "train.parquet")
        if not os.path.isfile(src_train):
            continue
        n_train = _parquet_rows(src_train)
        if n_train is None:
            continue
        bench = bench_hint or _bench_for_train_rows(n_train)
        if bench is None:
            print(f"[migrate] skip {src_text}: train rows={n_train} unknown bench")
            continue
        dst_text = os.path.join(mem_root, bench, "text")
        if os.path.isfile(os.path.join(dst_text, "train.parquet")) and os.path.isfile(
            os.path.join(dst_text, "test.parquet")
        ):
            print(f"[migrate] bench target already exists: {dst_text}")
            continue
        print(f"[migrate] {src_text} (rows={n_train}) -> {dst_text}")
        _copy_pair(src_text, dst_text, dry_run=args.dry_run)
        copied += 1

    if copied == 0:
        print("[migrate] no migratable legacy parquet found (targets may already exist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
