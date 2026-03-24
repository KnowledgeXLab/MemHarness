#!/usr/bin/env python3
"""
对 ``_dump_validation_trajectories_jsonl`` 产出的 JSONL 轨迹调用 LLM，做失败归因分类（预实验）。

标签（**多标签**；共 **3** 类 + 可与多类并存）：
  - query_failure: 检索/查询不当，未召回到应召回的记忆或召回了明显无关/有害条目。
  - adaptation_failure: **记忆使用/情境侧**问题：注入内容所依赖的前提与当前情境不符、和/或仍以**绑定旧状态**的固化形式出现，agent **难以转化为当前有效行动**。
  - non_memory_failure: 指令/规划/动作格式/探索等非记忆因素**显著**促成失败（可与 query/adaptation 并存）。

模型必须只输出上述三类的 ``labels``（字符串数组，可多选）。汇总为 **multihot** 与 **常见组合**。

默认只处理 **失败** 轨迹（``episode_reward < --reward_threshold``）。加 ``--include_success`` 可连成功轨迹一起评。
可用 ``--only_with_memory`` 只分析 **memory_retrieval_count > 0** 的轨迹（常与失败筛选联用）。

"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# 注入格式与 ``agent_system/memory/memory_manager.py`` 的 ``_format_memory_prompt`` 一致：``memory 1:`` …（通常 top_k=3）
_MEMORY_LINE_RE = re.compile(r"^memory\s+(\d+)\s*:\s*(.*)$", re.MULTILINE)

_DEFAULT_OPENAI_BASE_URL = "http://35.220.164.252:3888/v1/"
_DEFAULT_SECRETS_PATH = Path(__file__).resolve().parent / "llm_secrets.json"

import httpx
from openai import OpenAI
from tqdm import tqdm


VALID_LABELS = frozenset({"query_failure", "adaptation_failure", "non_memory_failure"})


SYSTEM_PROMPT = """You are an expert analyst for memory-augmented interactive agents (e.g., ALFWorld-style text environments).

Your job is to read ONE episode: the **full text the policy saw at each step** and the **policy's text output**, then assign **one or more** failure-factor labels from the allowed set. Use **multi-label** whenever several factors clearly contributed (e.g., bad planning **and** misleading injected memory).

Important:
- Base your judgment **only on the episode text** below. Do not invent unseen environment facts.
- The INPUT at each step may bundle: task/instruction, current observation, chat history, and **optional injected retrieved-memory passages** (wording depends on the system). Use those passages when reasoning about memory.
- The episode text may be **truncated** (first + last turns only) to fit context; if so, a note appears at the top of the transcript. Still do your best, and weight evidence from **later visible steps** more heavily when judging the final failure.
- **labels** must be a **non-empty** JSON array of **distinct** strings from the allowed set: **query_failure**, **adaptation_failure**, **non_memory_failure**. Include **every** label for which you have **clear** supporting evidence (typically 1–2; at most 3).
- **adaptation_failure** (memory **after retrieval**): use when injected memories **hurt or fail to help** because they are **misaligned with the current situation** and/or **not usable as-is**—including: (a) **preconditions** or implicit assumptions (use authoring metadata when provided) **clash** with current INPUT observations; (b) content is **too tied to a past episode wording** or **not operationalizable** into correct admissible actions. **Do not** split this into finer subtypes; one label covers both.
- **query_failure**: use when retrieval/query clearly returned **irrelevant** passages relative to the step's goal/query, or when **obviously better** memories should have been retrieved (only if strongly inferable from the text).
- **non_memory_failure**: use when **instruction misunderstanding, invalid action format, poor exploration, or planning** clearly hurt success. You **may** combine it with **query_failure** and/or **adaptation_failure** if **both** apply.
- Do **not** add a label without evidence; do **not** output duplicate entries in **labels**.
- Output valid JSON only, no markdown fences, no extra keys beyond the schema.
"""


USER_PROMPT_TEMPLATE = """## Taxonomy (multi-label: set ``labels`` to all that apply; **three** label types only)

1) "query_failure"
   Retrieval / query mismatch: **relevant** memories were not retrieved, or **clearly wrong** items were retrieved, in a way that plausibly hurt the episode (infer from INPUT).

2) "adaptation_failure"
   **After retrieval**, injected content is a poor fit for **current** decision-making: e.g. **preconditions** or implicit assumptions conflict with **current** observations (see authoring metadata when present), and/or the text stays **too bound to an old situation** / **not operationalizable** so the agent fails to derive correct actions. **Covers both** "wrong-for-this-world" and "right-ish but unusable as pasted"; do **not** use separate labels for those subcases.

3) "non_memory_failure"
   **Non-memory** factors (instruction, format, exploration, planning) **clearly** contributed. May co-occur with **query_failure** and/or **adaptation_failure**.

## How to read the episode

The **episode transcript** is the primary evidence. It is a step-by-step log:
- **INPUT** = everything the policy receives at that step (observation + dialogue + any memory injection text).
- **OUTPUT** = the policy's textual reply / action.

Do **not** rely on any external IDs, training step numbers, dataset indices, or scalar aggregate rewards — they are not provided on purpose.

## Authoring metadata for injected memories (from memory bank JSONL)

The following blocks are **not** shown to the agent at runtime; they are **offline authoring fields** (subgoal, preconditions, why_useful) for each **injected** ``memory k: ...`` line parsed from INPUT, matched by string lookup against the memory bank.

{memory_authoring_context}

## Episode transcript

{episode_transcript}

## Required output JSON schema

Return a JSON object with EXACTLY these keys:
{{
  "labels": ["<one or more distinct strings from: query_failure, adaptation_failure, non_memory_failure>"],
  "confidence": <float in [0,1]>,
  "short_rationale": "<2-6 sentences, English or Chinese ok; explain multiple labels if more than one>",
  "key_evidence": "<short quote or paraphrase from the episode text>",
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
    p.add_argument(
        "--secrets",
        type=str,
        default=str(_DEFAULT_SECRETS_PATH),
        help="JSON file with optional api_key and base_url (default: scripts/llm_secrets.json).",
    )
    p.add_argument(
        "--base_url",
        default=None,
        help=f"OpenAI-compatible API base URL (else from secrets file, else {_DEFAULT_OPENAI_BASE_URL!r}).",
    )
    p.add_argument("--model", default="gpt-5-mini", help="Judge model name.")
    p.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key (else from secrets file or OPENAI_API_KEY).",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max_workers", type=int, default=20, help="Concurrent LLM calls.")
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
        default=True,
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
        default=800000,
        help="Hard cap on episode transcript characters (after turn selection); tail is cut if exceeded.",
    )
    p.add_argument(
        "--memory_records_jsonl",
        default="data/AgentGym/AgentTraj-L/alfworld_train_memory_records-gpt-5.1.jsonl",
        help="Memory bank JSONL; build index on memory_text for lookup. Empty string disables.",
    )
    p.add_argument(
        "--memory_meta_max_chars",
        type=int,
        default=100000,
        help="Max characters for the authoring-metadata section (0 = unlimited).",
    )
    p.add_argument(
        "--memory_fuzzy_match",
        action="store_true",
        help="If set, fall back to substring match (normalized) when exact memory_text key misses.",
    )
    return p.parse_args()


def _load_secrets_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_openai_credentials(args: argparse.Namespace) -> None:
    """Merge --api_key / --base_url with secrets JSON and env (OPENAI_API_KEY)."""
    sec_path = Path(args.secrets).expanduser().resolve()
    data = _load_secrets_json(sec_path)
    api_key = args.api_key or data.get("api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or data.get("base_url") or _DEFAULT_OPENAI_BASE_URL
    args.api_key = api_key
    args.base_url = base_url


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


def parse_and_validate_judge_labels(judged: dict[str, Any]) -> tuple[list[str] | None, str | None]:
    """Require non-empty ``labels``: only ``query_failure`` | ``adaptation_failure`` | ``non_memory_failure``."""
    labs = judged.get("labels")
    if labs is None:
        return None, "missing required `labels` array"
    if not isinstance(labs, list):
        return None, f"`labels` must be a list, got {type(labs).__name__}"
    out: list[str] = []
    seen: set[str] = set()
    for i, x in enumerate(labs):
        if not isinstance(x, str):
            return None, f"labels[{i}] must be string, got {type(x).__name__}"
        s = x.strip()
        if s not in VALID_LABELS:
            return None, f"invalid label in labels[{i}]: {x!r} (must be one of {sorted(VALID_LABELS)})"
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if not out:
        return None, "`labels` is empty (or only duplicates)"
    return out, None


def extract_labels_for_summary(raw: dict[str, Any] | None) -> list[str]:
    if not isinstance(raw, dict):
        return []
    labs, err = parse_and_validate_judge_labels(raw)
    return labs if not err and labs else []


def normalize_memory_text_key(text: str) -> str:
    """Match keys between injected ``memory k:`` lines and JSONL ``memory_text``."""
    return " ".join((text or "").strip().split())


def parse_injected_memory_lines(input_text: str) -> list[tuple[int, str]]:
    """Parse ``memory_manager._format_memory_prompt`` lines: ``memory 1: ...`` (top_k typically 3)."""
    out: list[tuple[int, str]] = []
    for m in _MEMORY_LINE_RE.finditer(input_text or ""):
        try:
            rank = int(m.group(1))
        except ValueError:
            continue
        body = (m.group(2) or "").strip()
        if body:
            out.append((rank, body))
    out.sort(key=lambda x: x[0])
    return out


def load_memory_bank_text_index(path: str) -> tuple[dict[str, dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """``normalized memory_text`` -> first JSON row; also return list for optional fuzzy match."""
    exact: dict[str, dict[str, Any]] = {}
    pairs: list[tuple[str, dict[str, Any]]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            mt = rec.get("memory_text")
            if not isinstance(mt, str) or not mt.strip():
                continue
            k = normalize_memory_text_key(mt)
            pairs.append((k, rec))
            if k not in exact:
                exact[k] = rec
    return exact, pairs


def lookup_memory_record(
    injected_line: str,
    exact: dict[str, dict[str, Any]],
    pairs: list[tuple[str, dict[str, Any]]],
    *,
    use_fuzzy: bool,
) -> tuple[dict[str, Any] | None, str]:
    k = normalize_memory_text_key(injected_line)
    if not k:
        return None, "empty_injection"
    if k in exact:
        return exact[k], "exact"
    if not use_fuzzy:
        return None, "miss"
    best: dict[str, Any] | None = None
    best_key_len = -1
    for bk, rec in pairs:
        if not bk:
            continue
        if bk in k or k in bk:
            if len(bk) > best_key_len:
                best = rec
                best_key_len = len(bk)
    if best is not None:
        return best, "fuzzy_substring"
    return None, "miss"


def build_memory_authoring_context_from_injections(
    turns: list[dict[str, Any]],
    exact: dict[str, dict[str, Any]],
    pairs: list[tuple[str, dict[str, Any]]],
    *,
    use_fuzzy: bool,
    max_total_chars: int,
    per_field_max: int = 1400,
) -> tuple[str, dict[str, Any]]:
    """Scan every turn INPUT for ``memory k:`` lines; match JSONL by memory_text string."""
    dbg: dict[str, Any] = {
        "n_injected_lines": 0,
        "n_unique_injected": 0,
        "match_exact": 0,
        "match_fuzzy": 0,
        "miss": 0,
        "injection_turns": [],
    }
    if not exact and not pairs:
        return (
            "(Memory bank index is empty or not loaded; cannot attach authoring metadata.)\n",
            dbg,
        )

    # unique normalized injection -> list of (turn_idx, rank, raw_snippet, match)
    seen: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []

    ordered_turns = sorted(turns, key=_safe_turn_index) if turns else []
    for t in ordered_turns:
        turn_idx = _safe_turn_index(t)
        inp = str(t.get("input", ""))
        for rank, snippet in parse_injected_memory_lines(inp):
            dbg["n_injected_lines"] += 1
            nk = normalize_memory_text_key(snippet)
            if not nk:
                continue
            rec, how = lookup_memory_record(snippet, exact, pairs, use_fuzzy=use_fuzzy)
            if how == "exact":
                dbg["match_exact"] += 1
            elif how == "fuzzy_substring":
                dbg["match_fuzzy"] += 1
            elif how == "miss":
                dbg["miss"] += 1
            if nk not in seen:
                seen[nk] = {
                    "snippet": snippet,
                    "turn_ranks": [],
                    "record": rec,
                    "how": how,
                }
                ordered_keys.append(nk)
            seen[nk]["turn_ranks"].append((turn_idx, rank))
            _how_rank = {"miss": 0, "empty_injection": 0, "fuzzy_substring": 1, "exact": 2}
            cur = seen[nk]["how"]
            if _how_rank.get(how, 0) > _how_rank.get(cur, 0):
                seen[nk]["record"] = rec
                seen[nk]["how"] = how
            elif rec is not None and seen[nk]["record"] is None:
                seen[nk]["record"] = rec
                seen[nk]["how"] = how

    dbg["n_unique_injected"] = len(ordered_keys)
    dbg["injection_turns"] = [
        {"normalized_key_preview": k[:80], "turn_ranks": seen[k]["turn_ranks"], "match": seen[k]["how"]}
        for k in ordered_keys
    ]

    if not ordered_keys:
        return (
            "(No ``memory 1:`` / ``memory 2:`` … lines were found in any INPUT; nothing to match.)\n",
            dbg,
        )

    def trunc(s: str, n: int) -> str:
        s = str(s).strip()
        if n <= 0 or len(s) <= n:
            return s
        return s[: n - 25] + "\n[... truncated ...]\n"

    parts: list[str] = []
    body_len = 0
    for nk in ordered_keys:
        info = seen[nk]
        tr = info["turn_ranks"]
        loc = ", ".join(f"turn#{ti + 1}:rank{r}" for ti, r in tr[:8])
        if len(tr) > 8:
            loc += ", ..."
        head = f"### Injected memory (as in INPUT) @ {loc}\n"
        head += f"- **text**: {trunc(info['snippet'], 2500)}\n"
        head += f"- **bank_lookup**: {info['how']}\n"
        rec = info["record"]
        if rec is None:
            chunk = head + "- **metadata**: *(no matching row in memory JSONL)*\n\n"
        else:
            # 只给 judge 与「前提是否成立 / 是否该用这条经验」相关的短字段；不写库构建时的 id、轨迹位置、长 state 等
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            chunk = head
            has_any = bool(
                (meta.get("subgoal") or "").strip()
                or (meta.get("preconditions") or "").strip()
                or (meta.get("why_useful") or "").strip()
            )
            if has_any:
                if (meta.get("subgoal") or "").strip():
                    chunk += f"- **subgoal** (authored): {trunc(str(meta.get('subgoal', '')), per_field_max)}\n"
                if (meta.get("preconditions") or "").strip():
                    chunk += f"- **preconditions** (authored): {trunc(str(meta.get('preconditions', '')), per_field_max)}\n"
                if (meta.get("why_useful") or "").strip():
                    chunk += f"- **why_useful** (authored): {trunc(str(meta.get('why_useful', '')), per_field_max)}\n"
            else:
                chunk += "- *(Matched memory bank row but no subgoal/preconditions/why_useful in metadata.)*\n"
            chunk += "\n"
        if max_total_chars > 0 and body_len + len(chunk) > max_total_chars:
            parts.append("\n[... additional injected memories omitted: memory_meta_max_chars ...]\n")
            break
        parts.append(chunk)
        body_len += len(chunk)

    return "".join(parts), dbg


def attach_memory_text_index(args: argparse.Namespace) -> None:
    args.memory_text_exact = {}
    args.memory_text_pairs = []
    raw = (getattr(args, "memory_records_jsonl", None) or "").strip()
    if not raw:
        return
    mp = Path(raw)
    if not mp.is_file():
        print(f"mem_fail_judge: warning: --memory_records_jsonl not a file ({mp}); metadata disabled.", file=sys.stderr)
        return
    args.memory_text_exact, args.memory_text_pairs = load_memory_bank_text_index(str(mp))
    print(
        f"mem_fail_judge: loaded {len(args.memory_text_exact)} unique memory_text keys from {mp}",
        file=sys.stderr,
    )


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


def default_output_path(input_path: str, args: argparse.Namespace) -> Path:

    if args.only_with_memory:
        out_path = input_path.replace(".jsonl", ".mem_judge-only-mem.jsonl")
    else:
        out_path = input_path.replace(".jsonl", ".mem_judge-all.jsonl")
    
    if args.include_success:
        out_path = out_path.replace(".jsonl", "_trajs.jsonl")
    else:
        out_path = out_path.replace(".jsonl", "_fail_trajs-1.jsonl")

    return Path(out_path)


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

    mem_ctx, mem_dbg = build_memory_authoring_context_from_injections(
        turns,
        getattr(args, "memory_text_exact", {}) or {},
        getattr(args, "memory_text_pairs", []) or [],
        use_fuzzy=bool(getattr(args, "memory_fuzzy_match", False)),
        max_total_chars=int(getattr(args, "memory_meta_max_chars", 0) or 0),
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        memory_authoring_context=mem_ctx,
        episode_transcript=episode_transcript,
    )

    result: dict[str, Any] = {
        "traj_uid": rec.get("traj_uid"),
        "data_source": rec.get("data_source"),
        "global_step": rec.get("global_step"),
        "episode_reward": rec.get("episode_reward"),
        "memory_retrieval_count": rec.get("memory_retrieval_count"),
        "num_turns": rec.get("num_turns"),
        "transcript_truncation": trunc_meta,
        "memory_lookup_debug": mem_dbg,
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
        labels_norm, label_err = parse_and_validate_judge_labels(judged)
        if label_err:
            result["error"] = label_err
            result["judge_raw"] = judged
        else:
            judged_out = {**judged, "labels": labels_norm}
            result["judge_labels"] = labels_norm
            result["judge_raw"] = judged_out
    except Exception as e:
        result["error"] = repr(e)
    return result


def summarize_counts(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Multihot: each trajectory can contribute to multiple label counts."""
    inclusion: Counter[str] = Counter()
    pattern: Counter[str] = Counter()
    n_err = 0
    for r in results:
        if r.get("error"):
            n_err += 1
            continue
        raw = r.get("judge_raw")
        labs = extract_labels_for_summary(raw if isinstance(raw, dict) else None)
        if not labs:
            n_err += 1
            continue
        for lb in labs:
            inclusion[lb] += 1
        pattern["+".join(sorted(labs))] += 1
    return {
        "total": len(results),
        "error_count": n_err,
        "label_inclusion_counts": dict(inclusion),
        "label_pattern_top": dict(pattern.most_common(40)),
    }


def main() -> None:
    args = parse_args()
    resolve_openai_credentials(args)
    attach_memory_text_index(args)
    rows = _iter_jsonl(args.input)
    selected = select_rows(rows, args)
    if args.max_trajectories > 0:
        selected = selected[: args.max_trajectories]

    out_path = args.output or default_output_path(args.input, args)
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

    if not args.api_key:
        sys.exit(
            "Missing API key: add api_key to scripts/llm_secrets.json, "
            "or pass --api_key, or set OPENAI_API_KEY."
        )

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
    print(json.dumps({"output": str(out_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
