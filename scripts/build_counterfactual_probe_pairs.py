#!/usr/bin/env python3
"""Build counterfactual adaptor probe pairs from AgentTraj-L memory JSONL + raw trajectories.

For each sampled memory record:
  - ``s_old``  = memory ``state_text`` (retrieval key written at extraction time)
  - ``p_old``  = memory ``memory_text``
  - ``task``   = from ``alfworld_train.json`` (``Your task is to: ...``)
  - ``s_plus`` = raw AlfWorld observation at ``source_step`` (ShareGPT human turn)
  - ``s_minus`` = LLM minimal counterfactual edit of ``s_plus`` (memory no longer applies)

Output JSONL is consumed by ``scripts/run_counterfactual_probe.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_system.memory.types import MEMORY_STATE_UNAVAILABLE_PLACEHOLDER

# --- Edit these defaults ---
domain = "webshop"  # webshop or alfworld 
DEFAULT_MEMORY_JSONL = f"data/MemAdaptor/AgentTraj-L/{domain}_train_memory_records-gpt-5.1.jsonl"
DEFAULT_ALFWORLD_TRAIN_JSON = f"data/MemAdaptor/AgentTraj-L/{domain}_train.json"
DEFAULT_OUTPUT_JSONL = f"data/MemAdaptor/exp_results/{domain}/probe/counterfactual_pairs.jsonl"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 42

DEFAULT_LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
DEFAULT_LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_LLM_MODEL = os.environ.get("PROBE_LLM_MODEL", "gpt-5.1")

_ALFWORLD_TASK_RE = re.compile(r"Your task is to:\s*([^\n]+)", re.IGNORECASE)
_AVAILABLE_ACTIONS_RE = re.compile(r"\nAVAILABLE ACTIONS:.*", re.IGNORECASE | re.DOTALL)
_WEBSHOP_OBS_RE = re.compile(r"^'[^']*'(?: \[SEP\] '[^']*')*$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

MINUS_SYSTEM_PROMPT = """You create ONE counterfactual mismatched observation for a memory probe.

You are given a matched current observation (s_plus) where a retrieved memory applies.
Produce s_minus: s_plus with ONE minimal factual change so the memory NO LONGER applies.

Rules:
- Keep the same observation style as s_plus (AlfWorld or WebShop), including entity names, ids, and length.
- Change only one applicability-related fact (see memory preconditions / principle).
- Do NOT produce unnatural or gibberish text.
- Output valid JSON only (no markdown fences):
{"s_minus": "...", "flip": "one-line description of the single fact you changed"}
"""

WEBSHOP_MINUS_SYSTEM_PROMPT = """You create ONE counterfactual mismatched WebShop observation for a memory probe.

You are given a matched current observation (s_plus) where a retrieved memory applies.
Produce s_minus: s_plus with ONE minimal factual change so the memory NO LONGER applies.

WebShop format rules (strict):
- s_minus MUST look like a WebShop formatted observation: 'segment' [SEP] 'segment' [SEP] ...
- Every segment MUST be wrapped in single quotes.
- Preserve [SEP] separators exactly as in s_plus.
- Change only ONE segment/value (or remove/add one segment) so the memory no longer applies.
- Do NOT output plain prose (e.g. "NoSearch", "nothing available") without quoted [SEP] segments.
- Keep overall length and structure similar to s_plus.

Output valid JSON only (no markdown fences):
{"s_minus": "...", "flip": "one-line description of the single fact you changed"}
"""


USER_PROMPT_TEMPLATE = """Initial task (fixed):
{task}

Retrieved historical state s_old (fixed; NOT the current observation):
{s_old}

Memory principle p_old:
{p_old}

Matched current observation s_plus (fixed; do NOT rewrite):
{s_plus}

Memory metadata (for applicability):
{metadata_json}

Return JSON with exactly: s_minus, flip.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory_jsonl", default=DEFAULT_MEMORY_JSONL, help="AgentTraj-L memory records JSONL.")
    parser.add_argument(
        "--alfworld_train_json",
        default=DEFAULT_ALFWORLD_TRAIN_JSON,
        help="Raw AlfWorld trajectories (ShareGPT) for task + observations.",
    )
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT_JSONL, help="Output probe pairs JSONL.")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of pairs to build.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--llm_base_url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm_api_key", default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm_max_tokens", type=int, default=1024)
    parser.add_argument("--llm_temperature", type=float, default=0.6)
    parser.add_argument("--max_workers", type=int, default=64, help="Parallel LLM requests.")
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Only sample + write s_plus from trajectories (no s_minus); writes .partial.jsonl.",
    )
    parser.add_argument(
        "--exclude_step1_search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop WebShop pairs with source_step=1 and s_plus=='Search'.",
    )
    parser.add_argument(
        "--min_s_old_s_plus_jaccard",
        type=float,
        default=0.05,
        help="Minimum token Jaccard similarity between s_old and s_plus (filters large semantic gaps).",
    )
    parser.add_argument(
        "--llm_retries",
        type=int,
        default=3,
        help="Retries per pair when s_minus validation fails.",
    )
    return parser.parse_args()


def load_episode_index(path: str) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        episodes = json.load(handle)
    if not isinstance(episodes, list):
        raise ValueError(f"{path}: expected JSON list of episodes")
    index: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        item_id = str(ep.get("item_id") or "").strip()
        if item_id:
            index[item_id] = ep
    if not index:
        raise RuntimeError(f"No episodes with item_id in {path}")
    return index


def resolve_episode_id(row: dict[str, Any]) -> str:
    meta = row.get("metadata")
    if isinstance(meta, dict):
        did = str(meta.get("dataset_item_id") or "").strip()
        if did:
            return did
    return str(row.get("source_episode_id") or "").strip()


def _webshop_instruction_block_and_tail(raw_human: str) -> tuple[str, list[str]]:
    parts = raw_human.split(" [SEP] ")
    for i, part in enumerate(parts):
        if part.strip() != "Instruction:":
            continue
        if i + 1 >= len(parts):
            raise ValueError("Instruction: with no task segment")
        task = parts[i + 1].strip()
        tail = parts[i + 2 :]
        if not tail:
            raise ValueError("empty tail after instruction")
        return task, tail
    raise ValueError("no Instruction: segment in WebShop observation")


def _extract_webshop_task(raw_human: str) -> str:
    task, _tail = _webshop_instruction_block_and_tail(raw_human)
    return task


def _format_webshop_observation(raw_human: str) -> str:
    _task, tail = _webshop_instruction_block_and_tail(raw_human)
    return " [SEP] ".join(f"'{part}'" for part in tail).strip()


def _is_webshop_raw_obs(text: str) -> bool:
    return " [SEP] " in text and "Instruction:" in text


def task_from_episode(episode: dict[str, Any]) -> str:
    for turn in episode.get("conversations") or []:
        if not isinstance(turn, dict) or turn.get("from") != "human":
            continue
        raw = str(turn.get("value") or "").strip()
        if not raw:
            continue
        m = _ALFWORLD_TASK_RE.search(raw)
        if m:
            return " ".join(m.group(1).strip().split())
        if _is_webshop_raw_obs(raw):
            try:
                return _extract_webshop_task(raw)
            except ValueError:
                continue
    return ""


def raw_obs_at_source_step(episode: dict[str, Any], source_step: int) -> str:
    """Map 1-based agent action step -> adaptor-facing current observation at that step."""
    step = int(source_step)
    if step < 1:
        raise ValueError(f"source_step must be >= 1, got {source_step}")
    conv = episode.get("conversations") or []
    idx = 2 * step
    if idx >= len(conv):
        raise IndexError(f"source_step={step} -> conv[{idx}] out of range (len={len(conv)})")
    turn = conv[idx]
    if not isinstance(turn, dict) or turn.get("from") != "human":
        raise ValueError(f"source_step={step} -> conv[{idx}] is not a human observation")
    text = str(turn.get("value") or "").strip()
    if _is_webshop_raw_obs(text):
        text = _format_webshop_observation(text)
    else:
        text = _AVAILABLE_ACTIONS_RE.sub("", text).strip()
    if not text:
        raise ValueError(f"empty observation at source_step={step}")
    return text


def _is_valid_memory_row(row: dict[str, Any], episodes: dict[str, dict[str, Any]]) -> bool:
    state_text = str(row.get("state_text") or "").strip()
    memory_text = str(row.get("memory_text") or "").strip()
    if not state_text or not memory_text:
        return False
    if state_text in {"(none)", MEMORY_STATE_UNAVAILABLE_PLACEHOLDER}:
        return False
    if len(memory_text) < 20:
        return False
    ep_id = resolve_episode_id(row)
    if not ep_id or ep_id not in episodes:
        return False
    try:
        source_step = int(row.get("source_step"))
    except (TypeError, ValueError):
        return False
    try:
        raw_obs_at_source_step(episodes[ep_id], source_step)
    except (ValueError, IndexError):
        return False
    if not task_from_episode(episodes[ep_id]):
        return False
    return True


def _looks_like_webshop_obs(text: str) -> bool:
    t = str(text or "").strip()
    if not t.startswith("'"):
        return False
    if " [SEP] " in t:
        return bool(_WEBSHOP_OBS_RE.match(t))
    return t.endswith("'") and t.count("'") >= 2


def _is_webshop_probe_base(base: dict[str, Any]) -> bool:
    if str(base.get("task_name") or "").strip().lower() == "webshop":
        return True
    return _looks_like_webshop_obs(str(base.get("s_plus") or ""))


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _token_jaccard(left: str, right: str) -> float:
    a = _token_set(left)
    b = _token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _passes_semantic_gap(s_old: str, s_plus: str, *, min_jaccard: float) -> bool:
    return _token_jaccard(s_old, s_plus) >= min_jaccard


def _exclude_step1_search(base: dict[str, Any]) -> bool:
    if int(base.get("source_step") or 0) != 1:
        return False
    if not _is_webshop_probe_base(base):
        return False
    return str(base.get("s_plus") or "").strip() == "'Search'"


def _is_valid_webshop_observation(text: str) -> bool:
    return _looks_like_webshop_obs(text)


def _validate_s_minus(base: dict[str, Any], s_minus: str) -> None:
    s_minus = str(s_minus or "").strip()
    if not s_minus:
        raise ValueError("empty s_minus")
    if s_minus == str(base.get("s_plus") or "").strip():
        raise ValueError("s_minus equals s_plus")
    if _is_webshop_probe_base(base) and not _is_valid_webshop_observation(s_minus):
        raise ValueError("s_minus is not valid WebShop formatted observation")


def _load_valid_memory_rows(path: str, episodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _is_valid_memory_row(row, episodes):
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No valid memory rows in {path} (need resolvable episode + source_step obs)")
    return rows


def _collect_filtered_bases(
    memory_rows: list[dict[str, Any]],
    episodes: dict[str, dict[str, Any]],
    *,
    exclude_step1_search: bool,
    min_s_old_s_plus_jaccard: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    bases: list[dict[str, Any]] = []
    stats = {
        "valid_memory_rows": len(memory_rows),
        "excluded_step1_search": 0,
        "excluded_semantic_gap": 0,
        "eligible_bases": 0,
    }
    for row in memory_rows:
        base = _build_probe_base(row, episodes)
        if exclude_step1_search and _exclude_step1_search(base):
            stats["excluded_step1_search"] += 1
            continue
        if not _passes_semantic_gap(
            str(base["s_old"]),
            str(base["s_plus"]),
            min_jaccard=min_s_old_s_plus_jaccard,
        ):
            stats["excluded_semantic_gap"] += 1
            continue
        bases.append(base)
    stats["eligible_bases"] = len(bases)
    return bases, stats


def _sample_rows(path: str, sample_size: int, seed: int, episodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _load_valid_memory_rows(path, episodes)
    rng = random.Random(seed)
    if sample_size >= len(rows):
        sampled = list(rows)
        rng.shuffle(sampled)
    else:
        sampled = rng.sample(rows, sample_size)
    return sampled


def _build_probe_base(row: dict[str, Any], episodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ep_id = resolve_episode_id(row)
    episode = episodes[ep_id]
    source_step = int(row["source_step"])
    return {
        "memory_id": str(row.get("memory_id") or ""),
        "task_name": str(row.get("task_name") or "alfworld"),
        "dataset_item_id": ep_id,
        "task": task_from_episode(episode),
        "s_old": str(row.get("state_text") or "").strip(),
        "p_old": str(row.get("memory_text") or "").strip(),
        "s_plus": raw_obs_at_source_step(episode, source_step),
        "source_step": source_step,
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
    }


def _parse_minus_response(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty LLM response")
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON root must be an object")
    for key in ("s_minus", "flip"):
        if not str(obj.get(key, "")).strip():
            raise ValueError(f"missing or empty field: {key}")
    return obj


def _build_llm_prompt(base: dict[str, Any]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        task=base["task"],
        s_old=base["s_old"],
        p_old=base["p_old"],
        s_plus=base["s_plus"],
        metadata_json=json.dumps(base.get("metadata") or {}, ensure_ascii=False),
    )


def _minus_system_prompt(base: dict[str, Any]) -> str:
    if _is_webshop_probe_base(base):
        return WEBSHOP_MINUS_SYSTEM_PROMPT
    return MINUS_SYSTEM_PROMPT


def _call_llm(
    client: OpenAI,
    model: str,
    base: dict[str, Any],
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _minus_system_prompt(base)},
            {"role": "user", "content": _build_llm_prompt(base)},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = completion.choices[0].message.content or ""
    parsed = _parse_minus_response(content)
    s_minus = str(parsed["s_minus"]).strip()
    _validate_s_minus(base, s_minus)
    return {
        **parsed,
        "s_minus": s_minus,
    }


def _build_one_pair(
    base: dict[str, Any],
    client: OpenAI,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    llm_retries: int,
) -> dict[str, Any]:
    last_error = ""
    for _attempt in range(max(1, llm_retries)):
        try:
            parsed = _call_llm(
                client,
                model,
                base,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                **base,
                "s_minus": parsed["s_minus"],
                "flip": str(parsed["flip"]).strip(),
            }
        except Exception as exc:
            last_error = str(exc)
    return {"error": last_error or "unknown LLM/validation error", "memory_id": base.get("memory_id", "")}


def _select_bases_for_output(
    candidates: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(candidates) < sample_size:
        raise RuntimeError(
            f"Not enough filtered candidates ({len(candidates)}) for sample_size={sample_size}. "
            "Relax filters or increase memory pool."
        )
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled[:sample_size]


def main() -> None:
    args = parse_args()
    memory_jsonl = os.path.abspath(args.memory_jsonl)
    alfworld_json = os.path.abspath(args.alfworld_train_json)
    if not os.path.isfile(memory_jsonl):
        raise FileNotFoundError(f"memory_jsonl not found: {memory_jsonl}")
    if not os.path.isfile(alfworld_json):
        raise FileNotFoundError(f"alfworld_train_json not found: {alfworld_json}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
    episodes = load_episode_index(alfworld_json)
    print(f"Loaded {len(episodes)} episodes from {alfworld_json}", flush=True)

    memory_rows = _load_valid_memory_rows(memory_jsonl, episodes)
    candidates, filter_stats = _collect_filtered_bases(
        memory_rows,
        episodes,
        exclude_step1_search=args.exclude_step1_search,
        min_s_old_s_plus_jaccard=args.min_s_old_s_plus_jaccard,
    )
    print(json.dumps(filter_stats, ensure_ascii=False, indent=2), flush=True)

    if args.skip_llm:
        bases = _select_bases_for_output(candidates, sample_size=args.sample_size, seed=args.seed)
        print(f"Selected {len(bases)} filtered bases (seed={args.seed})", flush=True)
        partial_path = f"{os.path.abspath(args.output_jsonl)}.partial.jsonl"
        with open(partial_path, "w", encoding="utf-8") as handle:
            for base in bases:
                handle.write(json.dumps(base, ensure_ascii=False) + "\n")
        print(f"Wrote s_plus-only partial pairs -> {partial_path}", flush=True)
        return

    client = OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url)
    rng = random.Random(args.seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    results: list[dict[str, Any]] = []
    errors = 0
    cursor = 0
    wave = max(1, args.max_workers * 2)
    with tqdm(total=args.sample_size, desc="valid pairs") as pbar:
        while len(results) < args.sample_size and cursor < len(shuffled):
            batch = shuffled[cursor : cursor + wave]
            cursor += wave
            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
                futures = [
                    pool.submit(
                        _build_one_pair,
                        base,
                        client,
                        model=args.llm_model,
                        max_tokens=args.llm_max_tokens,
                        temperature=args.llm_temperature,
                        llm_retries=args.llm_retries,
                    )
                    for base in batch
                ]
                for fut in as_completed(futures):
                    rec = fut.result()
                    if "error" in rec:
                        errors += 1
                        continue
                    results.append(rec)
                    pbar.update(1)
                    if len(results) >= args.sample_size:
                        break

    if len(results) < args.sample_size:
        raise RuntimeError(
            f"Only built {len(results)} valid pairs (need {args.sample_size}). "
            f"LLM/validation errors={errors}. Relax filters or increase llm_retries."
        )

    results = results[: args.sample_size]

    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for rec in results:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "memory_jsonl": memory_jsonl,
                "alfworld_train_json": alfworld_json,
                "output_jsonl": os.path.abspath(args.output_jsonl),
                "requested": args.sample_size,
                "written_pairs": len(results),
                "llm_errors": errors,
                "filters": filter_stats,
                "exclude_step1_search": bool(args.exclude_step1_search),
                "min_s_old_s_plus_jaccard": float(args.min_s_old_s_plus_jaccard),
                "llm_retries": int(args.llm_retries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
