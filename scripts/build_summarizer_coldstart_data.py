#!/usr/bin/env python3
"""
Build cold-start SFT data for **experience summarization** (JSON memory extraction).

Trajectory text is built from **agent cold-start** parquet rows (``build_alfworld_coldstart_data.py`` /
``build_webshop_coldstart_data.py``): each step uses the MemAdaptor chat ``messages`` (env user prompts,
``<memory>`` injection, assistant outputs with ``<think>`` / ``<action>`` / ``<memory_retrieve>``).

Prompts match ``agent_system/memory/experience_summarizer.py``. Gold JSON labels come from teacher
``*_memory_records-*.jsonl``, joined on ``item_id`` (default group key ``metadata.dataset_item_id``).

By default, writes **merged** train/val: agent rows + summarizer rows (same split as input parquets).

Examples::

  python3 scripts/build_summarizer_coldstart_data.py --task alfworld \\
    --output-dir data/MemAdaptor/cold_start/alfworld/mixed_agent_summarizer_20260519

  python3 scripts/build_summarizer_coldstart_data.py --task webshop --schema compact \\
    --output-dir data/MemAdaptor/cold_start/webshop/mixed_agent_summarizer_20260519 \\
    --write-jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

from agent_system.memory.experience_summarizer import (
    DEFAULT_COMPACT_JSON_SYSTEM_PROMPT,
    DEFAULT_COMPACT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE,
    DEFAULT_JSON_SYSTEM_PROMPT,
    DEFAULT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE,
)

TASK_DEFAULTS: Dict[str, Dict[str, str]] = {
    "alfworld": {
        "task_name": "alfworld",
        "memory_jsonl": "data/MemAdaptor/AgentTraj-L/alfworld_train_memory_records-gpt-5.1.jsonl",
        "agent_train_parquet": "data/MemAdaptor/cold_start/alfworld/20260429_sample1000_seed42_train.parquet",
        "agent_val_parquet": "data/MemAdaptor/cold_start/alfworld/20260429_sample100_seed42_val.parquet",
    },
    "webshop": {
        "task_name": "webshop",
        "memory_jsonl": "data/MemAdaptor/AgentTraj-L/webshop_train_memory_records-gpt-5.1.jsonl",
        "agent_train_parquet": "data/MemAdaptor/cold_start/webshop/20260511_sample1000_seed42_train.parquet",
        "agent_val_parquet": "data/MemAdaptor/cold_start/webshop/20260511_sample100_seed42_val.parquet",
    },
}


def validate_memory_record(rec: Dict[str, Any], *, path_hint: str, line_no: int) -> None:
    required = ("memory_id", "source_step", "memory_text", "state_text")
    missing = [k for k in required if k not in rec]
    if missing:
        raise KeyError(f"{path_hint}:{line_no}: missing keys {missing}")
    int(rec["source_step"])


def iter_memory_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no}: expected object")
            validate_memory_record(obj, path_hint=path, line_no=line_no)
            yield obj


def resolve_memory_group_value(rec: Dict[str, Any], group_key: str) -> str:
    if group_key == "dataset_item_id":
        md = rec.get("metadata")
        if not isinstance(md, dict):
            raise KeyError("memory record missing metadata for dataset_item_id")
        v = md.get("dataset_item_id")
        if v is None or not str(v).strip():
            raise KeyError("metadata.dataset_item_id empty")
        return str(v).strip()
    v = rec.get(group_key)
    if v is None or not str(v).strip():
        raise KeyError(f"memory record missing {group_key!r}")
    return str(v).strip()


def group_memory_records(
    records: Sequence[Dict[str, Any]],
    group_key: str,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        out[resolve_memory_group_value(rec, group_key)].append(rec)
    for k in out:
        out[k].sort(key=lambda r: (int(r["source_step"]), str(r["memory_id"])))
    return out


def normalize_messages(cell: Any) -> List[Dict[str, str]]:
    """Parquet ``messages`` cell -> list of {role, content} dicts."""
    if cell is None:
        return []
    if hasattr(cell, "tolist"):
        cell = cell.tolist()
    if not isinstance(cell, list):
        raise TypeError(f"messages must be a list, got {type(cell).__name__}")
    out: List[Dict[str, str]] = []
    for i, msg in enumerate(cell):
        if not isinstance(msg, dict):
            raise TypeError(f"messages[{i}] must be dict, got {type(msg).__name__}")
        if "role" not in msg or "content" not in msg:
            raise KeyError(f"messages[{i}] missing role/content: keys={list(msg.keys())}")
        role = str(msg["role"]).strip().lower()
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"messages[{i}] unexpected role={role!r}")
        content = msg["content"]
        if content is None:
            content = ""
        out.append({"role": role, "content": str(content)})
    return out


def steps_from_agent_coldstart_messages(
    messages: Sequence[Dict[str, str]],
) -> List[Tuple[str, str]]:
    """
    One step per assistant turn (aligned with online summarizer rollout steps).

    Context = all user messages since the previous assistant turn (env observation, ``<memory>`` blocks, etc.).
    Action = raw assistant content (XML: think / action / memory_retrieve).
    """
    steps: List[Tuple[str, str]] = []
    pending_user_parts: List[str] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            continue
        if role == "user":
            if content:
                pending_user_parts.append(content)
            continue
        if role == "assistant":
            if not content:
                continue
            ctx = "\n\n".join(pending_user_parts)
            steps.append((ctx, content))
            pending_user_parts = []

    return steps


def compress_steps_head_tail(
    steps: List[Tuple[str, str]],
    *,
    max_turns: int,
    head_turns: int,
    tail_turns: int,
) -> List[Tuple[str, str]]:
    if max_turns <= 0 or len(steps) <= max_turns:
        return steps
    h = max(0, head_turns)
    t = max(0, tail_turns)
    if h + t >= len(steps):
        return steps
    head = steps[:h]
    tail = steps[-t:] if t > 0 else []
    mid_budget = max(0, max_turns - len(head) - len(tail))
    if mid_budget <= 0:
        return head + tail
    mid_start = h
    mid_end = len(steps) - t if t > 0 else len(steps)
    mid = steps[mid_start:mid_end]
    if len(mid) <= mid_budget:
        return head + mid + tail
    stride = max(1, len(mid) // mid_budget)
    mid_sampled = [mid[i] for i in range(0, len(mid), stride)][:mid_budget]
    return head + mid_sampled + tail


def format_trajectory_plain(
    steps: List[Tuple[str, str]],
    *,
    max_chars: int,
) -> str:
    lines: List[str] = []
    for i, (ctx, act) in enumerate(steps, start=1):
        lines.append(f"--- Step {i} ---")
        if ctx.strip():
            lines.append(f"Context (prompt):\n{ctx.strip()}")
        if act.strip():
            lines.append(f"Action (model output):\n{act.strip()}")
    body = "\n\n".join(lines)
    if max_chars > 0 and len(body) > max_chars:
        body = body[: max_chars - 80] + "\n\n[Trajectory truncated for cold-start prompt budget.]\n"
    return body


def memories_to_json_object(
    memories: Sequence[Dict[str, Any]],
    *,
    schema: str,
    max_memories: int,
) -> dict[str, Any]:
    schema_n = (schema or "compact").strip().lower()
    capped = list(memories)[: max(1, max_memories)]

    if schema_n == "compact":
        items: List[dict[str, Any]] = []
        for r in capped:
            mem = str(r.get("memory_text") or "").strip()
            if not mem:
                continue
            sit = str(r.get("state_text") or "").strip()
            if not sit:
                sit = "Situation grounded in the successful trajectory."
            items.append({"situation": sit, "memory": mem})
        return {"memories": items}

    items = []
    for r in capped:
        mem = str(r.get("memory_text") or "").strip()
        if not mem:
            continue
        st = str(r.get("state_text") or "").strip()
        at = str(r.get("action_text") or "").strip()
        if not st or not at:
            continue
        entry: dict[str, Any] = {
            "source_step": int(r["source_step"]),
            "state_text": st,
            "action_text": at,
            "memory_text": mem,
        }
        md = r.get("metadata")
        if isinstance(md, dict) and md:
            entry["metadata"] = md
        items.append(entry)
    return {"memories": items}


def build_summarizer_messages(
    *,
    task_name: str,
    trajectory_text: str,
    gold_memories: Sequence[Dict[str, Any]],
    schema: str,
    num_memories_cap: int,
) -> List[Dict[str, str]]:
    schema_n = (schema or "compact").strip().lower()
    if schema_n == "compact":
        system_prompt = DEFAULT_COMPACT_JSON_SYSTEM_PROMPT
        user_tmpl = DEFAULT_COMPACT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE
    else:
        system_prompt = DEFAULT_JSON_SYSTEM_PROMPT
        user_tmpl = DEFAULT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE

    num_cap = max(1, min(num_memories_cap, len(gold_memories) or num_memories_cap))
    user_msg = user_tmpl.format(
        num_memories=num_cap,
        task_name=task_name,
        trajectory_text=trajectory_text,
    )
    gold_obj = memories_to_json_object(gold_memories, schema=schema_n, max_memories=num_cap)
    assistant_msg = json.dumps(gold_obj, ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def read_parquet(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def ensure_agent_data_kind(df: pd.DataFrame) -> pd.DataFrame:
    if "data_kind" in df.columns:
        return df
    out = df.copy()
    out["data_kind"] = "agent"
    return out


def build_summarizer_rows_from_agent_df(
    agent_df: pd.DataFrame,
    *,
    memory_groups: Dict[str, List[Dict[str, Any]]],
    task_name: str,
    schema: str,
    num_memories_cap: int,
    trajectory_max_chars: int,
    trajectory_max_turns: int,
    trajectory_head_turns: int,
    trajectory_tail_turns: int,
    max_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = defaultdict(int)
    rows: List[Dict[str, Any]] = []
    subset = agent_df if max_rows <= 0 else agent_df.head(max_rows)

    for row_i, row in tqdm(subset.iterrows(), total=len(subset), desc="Agent rows", unit="row"):
        row_d = row.to_dict()

        raw_id = row_d.get("item_id")
        if raw_id is None:
            stats["skipped_no_item_id"] += 1
            continue
        item_id = str(raw_id).strip()
        if not item_id:
            stats["skipped_no_item_id"] += 1
            continue

        mg = memory_groups.get(item_id)
        if not mg:
            stats["skipped_no_teacher_memory"] += 1
            continue

        try:
            messages = normalize_messages(row_d.get("messages"))
            if not messages:
                stats["skipped_bad"] += 1
                continue
            steps = steps_from_agent_coldstart_messages(messages)
            if not steps:
                stats["skipped_bad"] += 1
                continue
            steps = compress_steps_head_tail(
                steps,
                max_turns=trajectory_max_turns,
                head_turns=trajectory_head_turns,
                tail_turns=trajectory_tail_turns,
            )
            traj = format_trajectory_plain(steps, max_chars=trajectory_max_chars)
            if not traj.strip():
                stats["skipped_bad"] += 1
                continue
            gold_obj = memories_to_json_object(mg, schema=schema, max_memories=num_memories_cap)
            if not gold_obj.get("memories"):
                stats["skipped_bad"] += 1
                continue
            summ_messages = build_summarizer_messages(
                task_name=task_name,
                trajectory_text=traj,
                gold_memories=mg,
                schema=schema,
                num_memories_cap=num_memories_cap,
            )
        except Exception as e:
            print(f"[skip] item_id={item_id!r} row={row_i}: {e}", file=sys.stderr)
            stats["skipped_bad"] += 1
            continue

        rows.append(
            {
                "messages": summ_messages,
                "item_id": item_id,
                "data_kind": "summarizer",
                "schema": schema,
                "num_teacher_memories": len(mg),
                "trajectory_turns": len(steps),
            }
        )
        stats["summarizer_rows"] += 1

    return rows, dict(stats)


def merge_agent_and_summarizer(
    agent_df: pd.DataFrame,
    summ_df: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """
    合并 agent 数据和 summarizer 数据，并进行整体 shuffle，会打乱 data_kind（比如 agent/summarizer）顺序。

    注意：经过 sample(frac=1.0) 打乱后，不同 data_kind 类型的数据会混排。
    """
    agent_df = ensure_agent_data_kind(agent_df)
    if summ_df.empty:
        return agent_df.reset_index(drop=True)
    parts = [agent_df, summ_df]
    merged = pd.concat(parts, ignore_index=True)
    # sample(frac=1.0) 会打乱所有行，包括不同 data_kind
    return merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: str,
    *,
    write_jsonl: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "val.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"Wrote train {len(train_df)} rows -> {train_path}")
    print(f"Wrote val   {len(val_df)} rows -> {val_path}")
    if write_jsonl:
        for split, df in (("train", train_df), ("val", val_df)):
            if len(df) == 0:
                continue
            jpath = os.path.join(out_dir, f"{split}.jsonl")
            with open(jpath, "w", encoding="utf-8") as jf:
                for _, row in df.iterrows():
                    obj = {k: row[k] for k in df.columns}
                    msgs = obj.get("messages")
                    if hasattr(msgs, "tolist"):
                        obj["messages"] = msgs.tolist()
                    jf.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            print(f"Wrote {len(df)} rows -> {jpath}")


def process_split(
    agent_parquet_path: str,
    *,
    memory_groups: Dict[str, List[Dict[str, Any]]],
    task_name: str,
    schema: str,
    num_memories_cap: int,
    trajectory_max_chars: int,
    trajectory_max_turns: int,
    trajectory_head_turns: int,
    trajectory_tail_turns: int,
    max_rows: int,
    merge_agent: bool,
    seed: int,
    split_name: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    agent_df = read_parquet(agent_parquet_path)
    summ_rows, stats = build_summarizer_rows_from_agent_df(
        agent_df,
        memory_groups=memory_groups,
        task_name=task_name,
        schema=schema,
        num_memories_cap=num_memories_cap,
        trajectory_max_chars=trajectory_max_chars,
        trajectory_max_turns=trajectory_max_turns,
        trajectory_head_turns=trajectory_head_turns,
        trajectory_tail_turns=trajectory_tail_turns,
        max_rows=max_rows,
    )
    summ_df = pd.DataFrame(summ_rows) if summ_rows else pd.DataFrame()
    print(
        f"[{split_name}] agent={len(agent_df)} summarizer={len(summ_df)} "
        f"from {agent_parquet_path} stats={stats}"
    )
    if merge_agent:
        return merge_agent_and_summarizer(agent_df, summ_df, seed=seed), stats
    return summ_df, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_DEFAULTS.keys()),
        help="Benchmark: default agent parquets, memory JSONL, and task_name for prompts.",
    )
    p.add_argument(
        "--agent-train-parquet",
        default="",
        help="Agent cold-start train.parquet (default: task preset).",
    )
    p.add_argument(
        "--agent-val-parquet",
        default="",
        help="Agent cold-start val.parquet (default: task preset).",
    )
    p.add_argument("--memory-jsonl", default="", help="Teacher memory records JSONL.")
    p.add_argument(
        "--memory-group-key",
        default="dataset_item_id",
        help="Join teacher JSONL -> agent row item_id.",
    )
    p.add_argument(
        "--schema",
        default="compact",
        choices=("compact", "full"),
        help="Match GRPO env.memory.experience_summarizer.schema.",
    )
    p.add_argument(
        "--num-memories-cap",
        type=int,
        default=5,
        help="Max memories in prompt and gold JSON.",
    )
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed when merging agent + summarizer.")
    p.add_argument("--output-dir", required=True, help="Write train.parquet / val.parquet here.")
    p.add_argument("--write-jsonl", action="store_true", help="Also write train.jsonl / val.jsonl.")
    p.add_argument(
        "--no-merge-agent",
        action="store_true",
        help="Write summarizer rows only (no concatenation with agent parquets).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Debug: process at most N rows per split (0 = all).",
    )
    p.add_argument(
        "--trajectory-max-chars",
        type=int,
        default=24000,
        help="Truncate trajectory plaintext (0 = no limit).",
    )
    p.add_argument(
        "--trajectory-max-turns",
        type=int,
        default=40,
        help="If more assistant turns, keep head+tail (0 = keep all).",
    )
    p.add_argument("--trajectory-head-turns", type=int, default=10)
    p.add_argument("--trajectory-tail-turns", type=int, default=10)
    args = p.parse_args(list(argv) if argv is not None else None)

    defaults = TASK_DEFAULTS[args.task]
    agent_train = args.agent_train_parquet.strip() or defaults["agent_train_parquet"]
    agent_val = args.agent_val_parquet.strip() or defaults["agent_val_parquet"]
    memory_jsonl = args.memory_jsonl.strip() or defaults["memory_jsonl"]
    task_name = defaults["task_name"]
    merge_agent = not args.no_merge_agent

    memory_groups = group_memory_records(list(iter_memory_jsonl(memory_jsonl)), args.memory_group_key)
    print(f"Loaded {sum(len(v) for v in memory_groups.values())} teacher memories for {len(memory_groups)} episodes")

    common_kw = dict(
        memory_groups=memory_groups,
        task_name=task_name,
        schema=args.schema,
        num_memories_cap=int(args.num_memories_cap),
        trajectory_max_chars=int(args.trajectory_max_chars),
        trajectory_max_turns=int(args.trajectory_max_turns),
        trajectory_head_turns=int(args.trajectory_head_turns),
        trajectory_tail_turns=int(args.trajectory_tail_turns),
        max_rows=int(args.max_rows),
        merge_agent=merge_agent,
        seed=int(args.seed),
    )

    train_df, train_stats = process_split(agent_train, split_name="train", **common_kw)
    val_df, val_stats = process_split(agent_val, split_name="val", **common_kw)

    if train_df.empty and val_df.empty:
        raise SystemExit("No output rows; check agent parquets and memory-jsonl alignment.")

    write_outputs(train_df, val_df, args.output_dir.strip(), write_jsonl=bool(args.write_jsonl))
    if "data_kind" in train_df.columns:
        print("Train data_kind counts:\n", train_df["data_kind"].value_counts().to_string())
    if "data_kind" in val_df.columns and len(val_df) > 0:
        print("Val data_kind counts:\n", val_df["data_kind"].value_counts().to_string())
    print(f"Train summarizer built: {train_stats} | Val summarizer built: {val_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
