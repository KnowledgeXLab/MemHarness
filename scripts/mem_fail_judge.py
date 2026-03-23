#!/usr/bin/env python3
"""
对 ``_dump_validation_trajectories_jsonl`` 产出的 JSONL 轨迹调用 LLM，做失败归因分类（预实验）。

标签：
  - query_failure: 检索意图/查询不当，未召回到真正相关记忆（或应检索而未检索到）。
  - experience_failure: 召回记忆表面相关，但依赖的环境条件与当前不一致，经验本身不适用。
  - usage_failure: 记忆并非完全无效，但以「绑定旧状态」的固化形式注入，agent 难以转化为有效行动。
  - non_memory_failure: 失败主要不由 memory 机制解释（指令理解、格式、探索噪声、环境偶然等）。

默认只处理 **失败** 轨迹（``episode_reward < --reward_threshold``）。加 ``--include_success`` 可连成功轨迹一起评。
可用 ``--only_with_memory`` 只分析 **memory_retrieval_count > 0** 的轨迹（常与失败筛选联用）。

用法示例::

    export OPENAI_API_KEY=...
    python scripts/mem_fail_judge.py --input path/to/0.jsonl --output out/mem_judge.jsonl

    # 仅失败且 memory_retrieval_count > 0
    python scripts/mem_fail_judge.py -i val.jsonl -o out.jsonl --only_with_memory

    # 先看 prompt 不调用 API
    python scripts/mem_fail_judge.py -i val.jsonl --dry_run --max_trajectories 2

长轨迹（如 50 步）可限制送入 judge 的步数，避免超出上下文::

    # 超过 32 步时只保留前 16 步 + 后 16 步，再按字符截断
    python scripts/mem_fail_judge.py -i val.jsonl -o out.jsonl \\
        --episode_max_turns 32 --episode_head_turns 16 --episode_tail_turns 16 \\
        --max_chars_episode 100000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm


VALID_LABELS = frozenset(
    {"query_failure", "experience_failure", "usage_failure", "non_memory_failure"}
)


SYSTEM_PROMPT = """You are an expert analyst for memory-augmented interactive agents (e.g., ALFWorld-style text environments).

Your job is to read ONE episode: the **full text the policy saw at each step** and the **policy's text output**, then classify WHY the episode failed **with respect to memory usage**, using exactly one primary label from the allowed set.

Important:
- Base your judgment **only on the episode text** below. Do not invent unseen environment facts.
- The INPUT at each step may bundle: task/instruction, current observation, chat history, and **optional injected retrieved-memory passages** (wording depends on the system). Use those passages when reasoning about memory.
- The episode text may be **truncated** (first + last turns only) to fit context; if so, a note appears at the top of the transcript. Still do your best, and weight evidence from **later visible steps** more heavily when judging the final failure.
- If you cannot find any plausible memory-related issue in the text, prefer non_memory_failure.
- Prefer non_memory_failure when the dominant issue is misunderstanding instructions, bad planning, wrong action syntax, or exploration noise — not memory content.
- Output valid JSON only, no markdown fences, no extra keys beyond the schema.
"""


USER_PROMPT_TEMPLATE = """## Taxonomy (choose exactly one primary_label)

1) "query_failure"
   Retrieval / query was mismatched so **relevant** experience was not retrieved, or clearly wrong items were retrieved, in a way that plausibly caused failure (infer from what appears in INPUT).

2) "experience_failure"
   Retrieved or injected content is **semantically plausible as advice** but its **preconditions** do not match the true current situation (state drift, wrong object/room/state). The content is misleading here.

3) "usage_failure"
   Injected content may be broadly relevant, but is **too tied to an old situation / not operationalizable** so the agent fails to turn it into correct actions (even if retrieval seemed OK).

4) "non_memory_failure"
   Failure is **primarily not** explained by memory (e.g., instruction misunderstanding, invalid action format, poor search). Memory may be absent or irrelevant in the text.

## How to read the episode

The next section is the **only** evidence. It is a step-by-step log:
- **INPUT** = everything the policy receives at that step (observation + dialogue + any memory injection text).
- **OUTPUT** = the policy's textual reply / action.

Do **not** rely on any external IDs, training step numbers, dataset indices, or scalar aggregate rewards — they are not provided on purpose.

{episode_transcript}

## Required output JSON schema

Return a JSON object with EXACTLY these keys:
{{
  "primary_label": "<one of: query_failure | experience_failure | usage_failure | non_memory_failure>",
  "confidence": <float in [0,1]>,
  "short_rationale": "<2-5 sentences, English or Chinese ok>",
  "key_evidence": "<short quote or paraphrase from the episode text that supports the label>",
  "secondary_notes": "<optional extra caveats; empty string if none>"
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM judge for memory-related failure taxonomy on validation trajectory JSONL."
    )
    p.add_argument("--input", "-i", default="data/exp_results/MemAdaptor/pre_exp/alfworld/Qwen2.5-1.5B-Instruct-with_agentic_memory-retrieve_memory_text/val_traj/0.jsonl", help="Path to validation trajectories .jsonl (one JSON per line).")
    p.add_argument(
        "--output",
        "-o",
        default="",
        help="Output .jsonl path (default: <input>.mem_judge.jsonl next to input).",
    )
    p.add_argument("--base_url", default="http://35.220.164.252:3888/v1/", help="OpenAI-compatible API base URL.")
    p.add_argument("--model", default="gpt-5.1", help="Judge model name.")
    p.add_argument("--api_key", default="sk-5QyBNRgeFFiX6sY1aooYjvtygjNelFW87I6ziXkE6mP6tVeH", help="OpenAI API key.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_workers", type=int, default=10, help="Concurrent LLM calls.")
    p.add_argument("--max_trajectories", type=int, default=-1, help="Cap number of trajectories to judge (-1 = all).")
    p.add_argument(
        "--reward_threshold",
        type=float,
        default=1.0,
        help="Treat episode as FAILED if episode_reward < this (ALFWorld success is often 1.0).",
    )
    p.add_argument(
        "--include_success",
        action="store_true",
        help="Also judge successful trajectories (default: only episode_reward < reward_threshold).",
    )
    p.add_argument(
        "--only_with_memory",
        default=False,
        type=bool,
        help="Only judge trajectories with memory_retrieval_count > 0.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print one prompt and exit without API calls.")
    p.add_argument(
        "--episode_max_turns",
        type=int,
        default=50,
        help="If total turns exceed this, keep only head+tail (0 = never truncate by turn count).",
    )
    p.add_argument(
        "--episode_head_turns",
        type=int,
        default=25,
        help="When truncating turns, number of earliest steps to keep.",
    )
    p.add_argument(
        "--episode_tail_turns",
        type=int,
        default=25,
        help="When truncating turns, number of latest steps to keep.",
    )
    p.add_argument(
        "--max_chars_episode",
        type=int,
        default=400000,
        help="Hard cap on episode transcript characters (after turn selection); tail is cut if exceeded.",
    )
    return p.parse_args()


def _iter_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def _safe_turn_index(turn_obj: dict[str, Any]) -> int:
    t = turn_obj.get("turn", 0)
    if isinstance(t, int):
        return t
    try:
        return int(t)
    except (TypeError, ValueError):
        return 0


def _turn_has_memory_retrieve_request(turn: dict[str, Any]) -> bool:
    """A turn that mentions memory retrieval must not be dropped by turn-truncation."""
    blob = (str(turn.get("output", "")) + "\n" + str(turn.get("input", ""))).lower()
    return "<memory_retrieve>" in blob or "</memory_retrieve>" in blob


def _turn_outputs_memory_retrieve(turn: dict[str, Any]) -> bool:
    """True if the policy's OUTPUT asks for retrieval (next turn's INPUT usually contains injected memory)."""
    return "<memory_retrieve>" in str(turn.get("output", "")).lower()


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _select_turns_for_prompt(
    ordered: list[dict[str, Any]],
    episode_max_turns: int,
    head_turns: int,
    tail_turns: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """If episode is long, keep earliest + latest turns; middle turns with <memory_retrieve> are never dropped."""
    n = len(ordered)
    meta: dict[str, Any] = {
        "total_turns_in_dump": n,
        "turns_included": n,
        "omitted_middle_turns": 0,
        "head_turns_kept": n,
        "tail_turns_kept": n,
        "truncated_by_turns": False,
        "protected_memory_retrieve_positions": [],
        "protected_memory_retrieve_turn_ids": [],
    }
    if episode_max_turns <= 0 or n <= episode_max_turns:
        return ordered, meta

    h = max(0, head_turns)
    t = max(0, tail_turns)
    if h + t >= n:
        return ordered, meta

    keep_pos: set[int] = set(range(h)) | set(range(n - t, n))
    protected_positions: list[int] = []
    protected_ids: list[Any] = []
    for i in range(h, n - t):
        if _turn_has_memory_retrieve_request(ordered[i]):
            keep_pos.add(i)
            protected_positions.append(i)
            protected_ids.append(ordered[i].get("turn", i))

    # Next step after a retrieve request typically carries returned memory in INPUT; do not drop it.
    for i in range(n - 1):
        if _turn_outputs_memory_retrieve(ordered[i]):
            keep_pos.add(i + 1)

    selected = [ordered[i] for i in sorted(keep_pos)]
    omitted = n - len(selected)

    meta["truncated_by_turns"] = omitted > 0
    meta["turns_included"] = len(selected)
    meta["omitted_middle_turns"] = omitted
    meta["head_turns_kept"] = h
    meta["tail_turns_kept"] = t
    meta["protected_memory_retrieve_positions"] = protected_positions
    meta["protected_memory_retrieve_turn_ids"] = protected_ids
    return selected, meta


def format_episode_for_judge(
    turns: list[dict[str, Any]],
    *,
    episode_max_turns: int,
    head_turns: int,
    tail_turns: int,
    max_chars: int,
) -> tuple[str, dict[str, Any]]:
    """Build per-step INPUT/OUTPUT text; optionally drop middle turns, then apply char cap."""
    if not turns:
        return "(empty episode — no steps)\n", {"empty": True}

    ordered = sorted(turns, key=_safe_turn_index)
    selected, meta = _select_turns_for_prompt(ordered, episode_max_turns, head_turns, tail_turns)

    lines: list[str] = []
    if meta.get("truncated_by_turns"):
        lines.append("## Note on coverage\n")
        extra = ""
        if meta.get("protected_memory_retrieve_positions"):
            extra = (
                " Middle steps that contain **<memory_retrieve>** are always kept; "
                "the **very next** step after a retrieve in OUTPUT is also kept (injected memory). "
                f"Protected middle indices (0-based in ordered dump): {meta['protected_memory_retrieve_positions']}. "
            )
        lines.append(
            f"This episode has **{meta['total_turns_in_dump']}** steps in the dump. "
            f"Only the **first {meta['head_turns_kept']}** and **last {meta['tail_turns_kept']}** steps are shown, "
            f"plus protected memory-retrieve steps; **{meta['omitted_middle_turns']}** steps are omitted overall. "
            f"{extra}"
            "Prioritize evidence from the **last included steps** when explaining failure.\n"
        )

    for t in selected:
        turn_idx = t.get("turn", 0)
        inp = str(t.get("input", "")).strip()
        out = str(t.get("output", "")).strip()
        lines.append(f"\n--- Step {turn_idx+1} ---\n")
        lines.append("### INPUT\n")
        lines.append(inp + "\n")
        lines.append("### OUTPUT\n")
        lines.append(out + "\n")

    text = "\n".join(lines)
    meta = {**meta, "truncated_by_chars": False, "max_chars": max_chars}
    if max_chars > 0 and len(text) > max_chars:
        meta["chars_before_char_trunc"] = len(text)
        text = text[:max_chars] + "\n\n[... truncated by max_chars_episode ...]\n"
        meta["truncated_by_chars"] = True
        meta["chars_kept"] = len(text)
    return text, meta


def build_client(base_url: str, api_key: str, timeout: int) -> OpenAI:
    http_client = httpx.Client(verify=False, trust_env=False, timeout=float(timeout))
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def call_judge(
    client: OpenAI,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    return extract_json_object(content)


def select_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in rows:
        reward = rec.get("episode_reward")
        try:
            reward_f = float(reward) if reward is not None else None
        except (TypeError, ValueError):
            reward_f = None

        mcount = rec.get("memory_retrieval_count")
        try:
            mcount_f = float(mcount) if mcount is not None else 0.0
        except (TypeError, ValueError):
            mcount_f = 0.0

        if args.only_with_memory and mcount_f <= 0:
            continue

        if not args.include_success:
            if reward_f is None:
                continue
            if reward_f >= args.reward_threshold:
                continue

        out.append(rec)
    return out


def default_output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(p.with_name(p.name + ".mem_judge.jsonl"))


def process_one(
    rec: dict[str, Any],
    args: argparse.Namespace,
    client: OpenAI | None,
) -> dict[str, Any]:
    turns = rec.get("turns") or []
    if not isinstance(turns, list):
        turns = []

    episode_transcript, trunc_meta = format_episode_for_judge(
        turns,
        episode_max_turns=args.episode_max_turns,
        head_turns=args.episode_head_turns,
        tail_turns=args.episode_tail_turns,
        max_chars=args.max_chars_episode,
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(episode_transcript=episode_transcript)

    result: dict[str, Any] = {
        "traj_uid": rec.get("traj_uid"),
        "data_source": rec.get("data_source"),
        "global_step": rec.get("global_step"),
        "episode_reward": rec.get("episode_reward"),
        "memory_retrieval_count": rec.get("memory_retrieval_count"),
        "num_turns": rec.get("num_turns"),
        "transcript_truncation": trunc_meta,
        "judge_model": args.model,
        "user_prompt": user_prompt,
        "judge_raw": None,
        "error": None,
    }

    if args.dry_run:
        result["judge_raw"] = {"skipped": "dry_run"}
        return result

    assert client is not None
    try:
        judged = call_judge(
            client=client,
            model=args.model,
            user_prompt=user_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        label = judged.get("primary_label")
        if isinstance(label, str) and label.strip() not in VALID_LABELS:
            result["error"] = f"invalid primary_label from model: {label!r}"
            result["judge_raw"] = judged
        else:
            result["judge_raw"] = judged
    except Exception as e:
        result["error"] = repr(e)
    return result


def summarize_counts(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels: list[str] = []
    for r in results:
        if r.get("error"):
            labels.append("__error__")
            continue
        raw = r.get("judge_raw")
        if isinstance(raw, dict) and raw.get("primary_label"):
            labels.append(str(raw["primary_label"]))
        else:
            labels.append("__error__")
    cnt = Counter(labels)
    return {"primary_label_counts": dict(cnt), "total": len(results)}


def main() -> None:
    args = parse_args()
    rows = _iter_jsonl(args.input)
    selected = select_rows(rows, args)
    if args.max_trajectories > 0:
        selected = selected[: args.max_trajectories]

    out_path = args.output or default_output_path(args.input)
    if Path(out_path).exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace.")


    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        if not selected:
            print("No trajectories selected after filters.")
            return
        one = process_one(selected[0], args, client=None)
        print(one["user_prompt"][:8000])
        print("\n... [dry_run: truncated print] ...\n")
        return

    client = build_client(args.base_url, args.api_key, args.timeout)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(process_one, rec, args, client) for rec in selected]
        for fut in tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="mem_fail_judge",
            unit="traj",
        ):
            results.append(fut.result())

    # stable order by traj_uid
    results.sort(key=lambda x: str(x.get("traj_uid", "")))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize_counts(results)
    print(json.dumps({"output": out_path, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
