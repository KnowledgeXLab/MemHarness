#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import sys
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS


THOUGHT_ACTION_RE = re.compile(
    r"(?is)^\s*(?:Thought:\s*(?P<thought>.*?))?\s*(?:Action:\s*(?P<action>[^\n]*))?\s*$"
)


def _strip_sharegpt(s: Any) -> str:
    if not isinstance(s, str):
        raise TypeError(f"expected str sharegpt value, got {type(s).__name__}")
    return s.strip()


def parse_thought_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = _strip_sharegpt(text)
    if not t:
        raise ValueError("empty GPT turn text")
    m = THOUGHT_ACTION_RE.match(t)
    if not m:
        raise ValueError(f"GPT turn does not match Thought/Action pattern:\n{t[:800]}")
    raw_th = m.group("thought")
    raw_act = m.group("action")
    thought = raw_th.strip() if isinstance(raw_th, str) else None
    action = raw_act.strip() if isinstance(raw_act, str) else None
    return thought, action


def format_assistant_xml(
    *,
    thought: Optional[str],
    action: Optional[str],
    think_open: str,
    think_close: str,
    action_open: str,
    action_close: str,
    empty_thought_fallback: str,
    lowercase_action: bool,
) -> str:
    if thought is None and action is None:
        raise ValueError("GPT turn missing both Thought and Action")
    if action is None:
        raise ValueError("GPT turn has Thought but missing Action (required for XML cold-start)")
    ac = action.strip()
    if lowercase_action:
        ac = ac.lower()
    if not ac:
        raise ValueError("GPT turn has empty Action after strip")
    if thought is None:
        th = empty_thought_fallback.strip()
        if not th:
            raise ValueError("empty_thought_fallback is empty but Thought is missing")
    else:
        th = thought.strip()
        if not th:
            raise ValueError("GPT turn has empty Thought after strip")
    return (
        f"{think_open}{th}{think_close}\n"
        f"{action_open}{ac}{action_close}"
    )


DEFAULT_RETRIEVAL_INSTRUCTION_PROMPT = (
    "If you want to retrieve memory before acting, output exactly one query wrapped by {open_tag} and {close_tag}. "
    "After memory is returned, you will act in the next response."
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_memory_record(rec: Dict[str, Any], *, path_hint: str, line_no: int) -> None:
    required = ("memory_id", "source_step", "memory_text", "state_text")
    missing = [k for k in required if k not in rec]
    if missing:
        raise KeyError(f"{path_hint}:{line_no}: missing keys {missing}, keys={list(rec.keys())}")
    if rec["memory_id"] is None:
        raise ValueError(f"{path_hint}:{line_no}: memory_id is None")
    if rec["source_step"] is None:
        raise ValueError(f"{path_hint}:{line_no}: source_step is None")
    try:
        int(rec["source_step"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path_hint}:{line_no}: source_step not int-like: {rec['source_step']!r}") from e


def iter_memory_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no}: JSON line must be an object, got {type(obj).__name__}")
            validate_memory_record(obj, path_hint=path, line_no=line_no)
            yield obj


def _nonempty_join_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def resolve_memory_group_value(rec: Dict[str, Any], group_key: str) -> str:
    if group_key == "dataset_item_id":
        md = rec.get("metadata")
        if not isinstance(md, dict):
            raise KeyError(
                f"memory record missing metadata dict for group key {group_key!r}: keys={list(rec.keys())}"
            )
        v = _nonempty_join_str(md.get("dataset_item_id"))
        if v is None:
            raise KeyError(
                f"memory record metadata.dataset_item_id missing or empty "
                f"(extract_memory_records.py output): keys={list(rec.keys())}"
            )
        return v
    v = _nonempty_join_str(rec.get(group_key))
    if v is not None:
        return v
    raise KeyError(f"memory record missing usable group key {group_key!r}: keys={list(rec.keys())}")


def group_memory_records(
    records: Sequence[Dict[str, Any]],
    group_key: str,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = resolve_memory_group_value(rec, group_key)
        out[key].append(rec)
    for k in list(out.keys()):
        out[k].sort(
            key=lambda r: (
                int(r["source_step"]),
                str(r["memory_id"]),
            )
        )
    return out


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: float = 120.0,
    ) -> None:
        base_url = base_url.strip().rstrip("/") + "/"
        http_client = httpx.Client(verify=False, trust_env=False, timeout=float(timeout_sec))
        self._client = OpenAI(
            api_key=api_key if api_key.strip() else "EMPTY",
            base_url=base_url,
            http_client=http_client,
        )
        self.model = model

    def chat_completion(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str:
        if not self.model.strip():
            raise RuntimeError("retrieve model name is empty (--retrieve-model)")
        completion = self._client.chat.completions.create(
            model=self.model.strip(),
            messages=messages,
            temperature=temperature,
        )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("Unexpected API response (no assistant message content)")
        return str(content).strip()


def heuristic_retrieve_query(*, task_hint: str, state_text: str, memory_text: str, max_chars: int) -> str:
    parts = []
    if task_hint:
        parts.append(task_hint.strip()[:200])
    if state_text:
        parts.append(state_text.strip()[: max_chars // 2])
    if memory_text:
        parts.append(memory_text.strip()[: max_chars // 2])
    q = " ".join(parts).strip()
    if len(q) > max_chars:
        q = q[: max_chars]
    if not q:
        raise ValueError("heuristic_retrieve_query produced empty query (missing task/state/memory text)")
    return q


def _is_sharegpt_instruction_banner(text: str) -> bool:
    s = text.lstrip()
    return bool(s) and s.startswith("You are web shopping")


def _instruction_block_and_tail(raw_human: str) -> Tuple[str, List[str]]:
    parts = raw_human.split(" [SEP] ")
    for i, p in enumerate(parts):
        if p.strip() != "Instruction:":
            continue
        if i + 1 >= len(parts):
            raise ValueError("Instruction: with no task segment")
        task = parts[i + 1].strip()
        tail = parts[i + 2 :]
        if not tail:
            raise ValueError("empty tail after instruction (no observation / actions)")
        return task, tail
    raise ValueError("no Instruction: segment in WebShop observation")


def extract_webshop_task(raw_human: str) -> str:
    task, _tail = _instruction_block_and_tail(raw_human)
    return task


def format_webshop_observation(raw_human: str) -> str:
    _task, tail = _instruction_block_and_tail(raw_human)
    return " [SEP] ".join(f"'{p}'" for p in tail).strip()


def infer_webshop_admissible_strings(raw_human: str) -> List[str]:
    _, tail_raw = _instruction_block_and_tail(raw_human)
    tail = [p.strip() for p in tail_raw]
    if not tail:
        raise ValueError("no admissible tail")
    if len(tail) == 1 and tail[0] == "Search":
        return ["search[<your query>]"]
    return [f"click[{t.lower()}]" for t in tail]


def _fmt_available_actions_block(admissible: Sequence[str]) -> str:
    return "\n".join(f"'{s}'," for s in admissible)


def _webshop_mem_history_block(
    completed: List[Tuple[str, str]],
    history_length: int,
) -> Tuple[str, int]:
    """Match ``SimpleMemory.fetch`` in ``agent_system/memory/memory.py``."""
    recent = completed[-history_length:] if history_length > 0 else []
    valid_len = len(recent)
    start_idx = len(completed) - valid_len
    lines: List[str] = []
    for j, (obs_text, act) in enumerate(recent):
        step_num = start_idx + j + 1
        lines.append(f"[Observation {step_num}: '{obs_text}', Action {step_num}: '{act}']")
    return "\n".join(lines).strip(), valid_len


def build_webshop_user_message(
    *,
    task_description: str,
    completed_history: List[Tuple[str, str]],
    history_length: int,
    current_observation: str,
    admissible: Sequence[str],
    retrieval_instruction: str,
    retrieve_open: str,
    retrieve_close: str,
) -> str:
    adm_block = _fmt_available_actions_block(admissible)
    td = task_description.strip() if task_description.strip() else "(unknown task)"
    if len(completed_history) == 0:
        body = WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=td,
            current_observation=current_observation,
            available_actions=adm_block,
        )
    else:
        ah, valid_len = _webshop_mem_history_block(completed_history, history_length)
        body = WEBSHOP_TEMPLATE.format(
            task_description=td,
            step_count=len(completed_history),
            history_length=valid_len,
            action_history=ah,
            current_step=len(completed_history) + 1,
            current_observation=current_observation,
            available_actions=adm_block,
        )
    if retrieval_instruction.strip():
        hint = retrieval_instruction.format(open_tag=retrieve_open, close_tag=retrieve_close)
        body = f"{body}\n\n{hint}"
    return body


def build_messages_for_episode(
    episode: Dict[str, Any],
    *,
    think_open: str,
    think_close: str,
    action_open: str,
    action_close: str,
    empty_thought_fallback: str,
    skip_initial_ok_gpt: bool,
    memory_group: Optional[List[Dict[str, Any]]],
    memory_steps: Optional[set[int]],
    retrieve_open: str,
    retrieve_close: str,
    memory_open: str,
    memory_close: str,
    adapted_header: str,
    adapted_label_tmpl: str,
    llm: Optional[OpenAICompatClient],
    retrieve_max_query_chars: int,
    sleep_between_llm_sec: float,
    memadaptor_user_template: bool,
    webshop_history_length: int,
    retrieval_instruction_prompt: str,
    lowercase_action: bool,
    retrieve_only_thinking: str,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    stats = {"turns": 0, "skipped_gpt": 0, "retrieve_inserted": 0, "llm_queries": 0, "heuristic_queries": 0}
    if "conversations" not in episode:
        raise KeyError("episode missing key 'conversations'")
    conv = episode["conversations"]
    if not isinstance(conv, list):
        raise TypeError("episode['conversations'] must be a list")
    if "item_id" not in episode:
        raise KeyError("episode missing key 'item_id'")
    item_id = str(episode["item_id"]).strip()
    if not item_id:
        raise ValueError("episode item_id is empty")

    messages: List[Dict[str, str]] = []
    memory_turn_idx = 0
    gpt_ord = 0
    task_hint = ""
    task_description = ""
    completed_history: List[Tuple[str, str]] = []
    last_human_formatted_obs: Optional[str] = None
    last_human_raw: Optional[str] = None

    instr = (retrieval_instruction_prompt or "").strip() or DEFAULT_RETRIEVAL_INSTRUCTION_PROMPT

    for turn in conv:
        if not isinstance(turn, dict):
            raise TypeError(f"conversation turn must be dict, got {type(turn).__name__}")
        if "from" not in turn or "value" not in turn:
            raise KeyError(f"turn missing from/value: keys={list(turn.keys())}")
        role = turn["from"]
        val = _strip_sharegpt(turn["value"])
        if role == "human":
            if memadaptor_user_template and _is_sharegpt_instruction_banner(val):
                continue
            last_human_raw = val
            if " [SEP] " in val and "Instruction:" in val:
                try:
                    task_description = extract_webshop_task(val)
                    task_hint = val
                except ValueError:
                    pass

            if memadaptor_user_template:
                if task_description:
                    try:
                        formatted_obs = format_webshop_observation(val)
                        adm = infer_webshop_admissible_strings(val)
                    except ValueError as e:
                        raise ValueError(f"webshop observation parse failed: {e}") from e
                    last_human_formatted_obs = formatted_obs
                    user_body = build_webshop_user_message(
                        task_description=task_description,
                        completed_history=completed_history,
                        history_length=max(0, int(webshop_history_length)),
                        current_observation=formatted_obs,
                        admissible=adm,
                        retrieval_instruction=instr,
                        retrieve_open=retrieve_open,
                        retrieve_close=retrieve_close,
                    )
                    messages.append({"role": "user", "content": user_body})
                else:
                    messages.append({"role": "user", "content": val})
            else:
                messages.append({"role": "user", "content": val})
            continue

        if role == "gpt":
            gpt_ord += 1
            if skip_initial_ok_gpt and gpt_ord == 1 and val.lower().startswith("ok."):
                stats["skipped_gpt"] += 1
                continue

            memory_turn_idx += 1
            thought, action = parse_thought_action(val)
            assistant_xml = format_assistant_xml(
                thought=thought,
                action=action,
                think_open=think_open,
                think_close=think_close,
                action_open=action_open,
                action_close=action_close,
                empty_thought_fallback=empty_thought_fallback,
                lowercase_action=lowercase_action,
            )
            memory_user_inj: Optional[str] = None
            if memory_group is not None and memory_steps is not None and memory_turn_idx in memory_steps:
                rec = next((r for r in memory_group if int(r["source_step"]) == memory_turn_idx), None)
                if rec is not None:
                    mem_txt = str(rec["memory_text"]).strip()
                    real_obs = last_human_formatted_obs or (
                        format_webshop_observation(last_human_raw) if last_human_raw else ""
                    )
                    query_text: Optional[str] = None
                    if llm is not None:
                        sys_prompt = (
                            "You produce exactly ONE plain-text retrieval query for a WebShop agent memory bank. "
                            "Do not output XML tags in your answer (no <memory_retrieve>). "
                            "Do not output comma-separated keywords; write one concise natural-language query. "
                            "The query should be something an agent could ask at the current decision point to retrieve "
                            "the provided teacher memory principle. "
                            "Follow the retrieval protocol described below.\n\n"
                            + instr.format(open_tag=retrieve_open, close_tag=retrieve_close)
                        )
                        user_prompt = (
                            f"Task goal:\n{task_description}\n\n"
                            f"Current observation:\n{real_obs}\n\n"
                            f"Teacher memory principle to retrieve:\n{mem_txt}\n\n"
                            "Write a retrieval query that would retrieve this memory for the current decision. "
                            "Focus on the immediate decision bottleneck (search vs click vs buy), not generic browsing.\n\n"
                            "Write only the retrieval query text."
                        )
                        try:
                            query_text = llm.chat_completion(
                                [
                                    {"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": user_prompt},
                                ]
                            )
                        except Exception as e:
                            raise ValueError(
                                f"retrieve LLM failed (skipping episode): item_id={item_id!r} "
                                f"source_step={memory_turn_idx}: {type(e).__name__}: {e}"
                            ) from e
                        stats["llm_queries"] += 1
                        if sleep_between_llm_sec > 0:
                            time.sleep(sleep_between_llm_sec)
                    else:
                        query_text = heuristic_retrieve_query(
                            task_hint=task_hint,
                            state_text=real_obs,
                            memory_text=mem_txt,
                            max_chars=retrieve_max_query_chars,
                        )
                        stats["heuristic_queries"] += 1

                    retrieve_body = query_text.strip()
                    retrieve_suffix = f"{retrieve_open}{retrieve_body}{retrieve_close}"
                    stats["retrieve_inserted"] += 1

                    inner_lines = []
                    hdr = adapted_header.strip()
                    if hdr:
                        inner_lines.append(hdr)
                    lab = adapted_label_tmpl.strip()
                    if lab:
                        inner_lines.append(lab.replace("{index}", "1").rstrip() + "\n" + mem_txt)
                    else:
                        inner_lines.append(mem_txt)
                    inner = "\n\n".join(inner_lines)
                    memory_user_inj = f"{memory_open}\n{inner}\n{memory_close}"

                    rt = (retrieve_only_thinking or "").strip()
                    if not rt:
                        rt = "I should retrieve relevant memory before choosing an action for this step."
                    retrieve_assistant = f"{think_open}{rt}{think_close}\n{retrieve_suffix}"
                    messages.append({"role": "assistant", "content": retrieve_assistant})
                    messages.append({"role": "user", "content": memory_user_inj})

            messages.append({"role": "assistant", "content": assistant_xml})
            stats["turns"] += 1
            if last_human_formatted_obs is None or last_human_raw is None:
                raise ValueError(f"missing preceding human turn before gpt turn in episode {item_id}")
            act_plain = action.strip() if action else ""
            if lowercase_action:
                act_plain = act_plain.lower()
            completed_history.append((last_human_formatted_obs, act_plain))
            last_human_raw = None
            last_human_formatted_obs = None
            continue

        raise ValueError(f"unknown ShareGPT role {role!r} in episode {item_id}")

    return messages, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--train-json",
        default="./data/MemAdaptor/cold_start/webshop/webshop_train.json",
        help="Path to webshop_train.json (list of episodes).",
    )
    p.add_argument(
        "--memory-jsonl",
        default="./data/MemAdaptor/cold_start/webshop/webshop_train_memory_records-gpt-5.1.jsonl",
        help="Optional memory records JSONL (gpt teacher).",
    )
    p.add_argument(
        "--memory-group-key",
        default="dataset_item_id",
        help="Join key to episode item_id (default metadata.dataset_item_id).",
    )
    p.add_argument("--memory-match-rate", type=float, default=0.5, help="Fraction of eligible assistant turns with retrieve+inject (0..1).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes.")
    p.add_argument("--output-dir", default="", help="Write train.parquet and val.parquet here.")
    p.add_argument(
        "--output",
        default="./data/MemAdaptor/cold_start/webshop/webshop_coldstart.parquet",
        help="Single combined parquet path (if set, ignores val split). Use \"\" with --output-dir for train/val split.",
    )
    p.add_argument("--write-jsonl", default=True, help="Also write JSONL next to parquet.")

    p.add_argument("--think-open-tag", default="<think>")
    p.add_argument("--think-close-tag", default="</think>")
    p.add_argument("--action-open-tag", default="<action>")
    p.add_argument("--action-close-tag", default="</action>")
    p.add_argument("--retrieve-open-tag", default="<memory_retrieve>")
    p.add_argument("--retrieve-close-tag", default="</memory_retrieve>")
    p.add_argument("--memory-open-tag", default="<memory>")
    p.add_argument("--memory-close-tag", default="</memory>")
    p.add_argument(
        "--adapted-prompt-header",
        default="Relevant past memories are provided below. Reuse them only if you think which ones are useful.",
    )
    p.add_argument(
        "--adapted-label-template",
        default="Adapted memory principle {index}:\n",
    )
    p.add_argument(
        "--retrieve-only-thinking",
        default=(
            "I should retrieve relevant memory before choosing an action for this step."
        ),
        help="Text inside <think> for the retrieve-only turn (must not include <action>).",
    )
    p.add_argument(
        "--empty-thought-fallback",
        default="(no explicit reasoning in source; choose a valid admissible action.)",
    )
    p.add_argument("--skip-initial-ok-gpt", action="store_true", default=True)
    p.add_argument(
        "--raw-sharegpt-user-text",
        action="store_true",
        help="Keep AgentTraj-L human strings verbatim (disable WEBSHOP_TEMPLATE prompts).",
    )
    p.add_argument(
        "--webshop-history-length",
        type=int,
        default=2,
        help="Matches trainer env.history_length for WEBSHOP_TEMPLATE action_history.",
    )
    p.add_argument("--retrieval-instruction-prompt", default="", help="Uses {open_tag}/{close_tag}; empty = default.")
    p.add_argument(
        "--lowercase-action",
        action="store_true",
        default=True,
        help="Lowercase action text to match webshop_projection (default: on).",
    )
    p.add_argument(
        "--no-lowercase-action",
        dest="lowercase_action",
        action="store_false",
        help="Keep original casing inside <action>.",
    )

    p.add_argument(
        "--retrieve-api-url",
        required=True,
        help="OpenAI-compatible API base URL (same as extract_memory_records --base_url), e.g. http://host:port/v1/",
    )
    p.add_argument(
        "--retrieve-api-key",
        required=True,         
        help="Bearer token for the OpenAI-compatible API (optional if gateway allows empty key).",
    )
    p.add_argument("--retrieve-model", default="gpt-5.1", help="Model name passed to chat.completions.")
    p.add_argument("--retrieve-timeout-sec", type=float, default=120.0)
    p.add_argument("--retrieve-sleep-sec", type=float, default=0.0)
    p.add_argument("--retrieve-max-query-chars", type=int, default=900)
    p.add_argument(
        "--retrieve-workers",
        type=int,
        default=32,
        help="Parallelism over episodes when generating retrieve queries (HTTP).",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    if int(args.retrieve_workers) < 1:
        raise SystemExit("--retrieve-workers must be >= 1")

    data = load_json(args.train_json)
    if not isinstance(data, list):
        raise SystemExit("train-json must be a JSON array of episodes")

    memory_groups: Dict[str, List[Dict[str, Any]]] = {}
    if args.memory_jsonl.strip():
        recs = list(iter_memory_jsonl(args.memory_jsonl))
        memory_groups = group_memory_records(recs, args.memory_group_key)

    llm: Optional[OpenAICompatClient] = None
    url_set = bool(args.retrieve_api_url.strip())
    model_set = bool(args.retrieve_model.strip())
    if url_set ^ model_set:
        raise SystemExit("Either both --retrieve-api-url and --retrieve-model must be set, or neither.")
    if url_set and model_set:
        llm = OpenAICompatClient(
            base_url=args.retrieve_api_url.strip(),
            api_key=args.retrieve_api_key.strip(),
            model=args.retrieve_model.strip(),
            timeout_sec=float(args.retrieve_timeout_sec),
        )

    rows: List[Dict[str, Any]] = []
    tot_stats = defaultdict(int)

    n_eps = len(data)
    if args.max_episodes and args.max_episodes > 0:
        n_eps = min(n_eps, args.max_episodes)

    slice_eps = data[:n_eps]

    def run_one_episode(ep_i: int, episode: Dict[str, Any], *, parallel_rng: bool) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
        skip_stats: Dict[str, int] = defaultdict(int)
        skip_stats["episodes_skipped"] = 1
        raw_id = episode.get("item_id")
        if raw_id is None:
            print(f"[skip episode] ep_index={ep_i}: missing item_id", file=sys.stderr)
            return None, skip_stats
        item_id = str(raw_id).strip()
        if not item_id:
            print(f"[skip episode] ep_index={ep_i}: empty item_id", file=sys.stderr)
            return None, skip_stats
        mg = memory_groups[item_id] if item_id in memory_groups else None
        memory_steps: Optional[set[int]] = None
        if mg:
            steps = sorted({int(r["source_step"]) for r in mg})
            eligible = [s for s in steps if s >= 1]
            k = max(1, int(round(len(eligible) * float(args.memory_match_rate))))
            k = min(k, len(eligible))
            if eligible and k > 0:
                if parallel_rng:
                    rng_ep = random.Random(args.seed ^ (ep_i * 0x9E3779B9))
                    memory_steps = set(rng_ep.sample(eligible, k))
                else:
                    memory_steps = set(random.sample(eligible, k))

        try:
            msgs, st = build_messages_for_episode(
                episode,
                think_open=args.think_open_tag,
                think_close=args.think_close_tag,
                action_open=args.action_open_tag,
                action_close=args.action_close_tag,
                empty_thought_fallback=args.empty_thought_fallback,
                skip_initial_ok_gpt=args.skip_initial_ok_gpt,
                memory_group=mg,
                memory_steps=memory_steps,
                retrieve_open=args.retrieve_open_tag,
                retrieve_close=args.retrieve_close_tag,
                memory_open=args.memory_open_tag,
                memory_close=args.memory_close_tag,
                adapted_header=args.adapted_prompt_header,
                adapted_label_tmpl=args.adapted_label_template,
                llm=llm,
                retrieve_max_query_chars=int(args.retrieve_max_query_chars),
                sleep_between_llm_sec=float(args.retrieve_sleep_sec),
                memadaptor_user_template=not bool(args.raw_sharegpt_user_text),
                webshop_history_length=int(args.webshop_history_length),
                retrieval_instruction_prompt=str(args.retrieval_instruction_prompt or ""),
                lowercase_action=bool(args.lowercase_action),
                retrieve_only_thinking=str(args.retrieve_only_thinking or ""),
            )
        except (ValueError, TypeError, KeyError) as e:
            print(f"[skip episode] item_id={item_id!r} ep_index={ep_i}: {type(e).__name__}: {e}", file=sys.stderr)
            return None, skip_stats
        if not msgs:
            return None, st
        row = {
            "messages": msgs,
            "item_id": item_id,
            "has_memory_records": bool(mg),
            "memory_records_used": int(memory_steps is not None),
        }
        return row, st

    workers = int(args.retrieve_workers)
    if workers <= 1:
        random.seed(args.seed)
        for ep_i, episode in tqdm(enumerate(slice_eps), total=n_eps, desc="Episodes", unit="ep"):
            row, st = run_one_episode(ep_i, episode, parallel_rng=False)
            for kk, vv in st.items():
                tot_stats[kk] += int(vv)
            if row is None:
                continue
            rows.append(row)
    else:
        pairs = [(ep_i, slice_eps[ep_i]) for ep_i in range(n_eps)]

        def _parallel_worker(pair: Tuple[int, Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
            ep_i, episode = pair
            return run_one_episode(ep_i, episode, parallel_rng=True)

        with tqdm(total=n_eps, desc="Episodes", unit="ep") as pbar:

            def _parallel_worker_counted(
                pair: Tuple[int, Dict[str, Any]],
            ) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
                try:
                    return _parallel_worker(pair)
                finally:
                    pbar.update(1)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                ordered = list(ex.map(_parallel_worker_counted, pairs))
        for row, st in ordered:
            for kk, vv in st.items():
                tot_stats[kk] += int(vv)
            if row is None:
                continue
            rows.append(row)

    if not rows:
        raise SystemExit("No rows produced (check input paths / parsing).")

    df = pd.DataFrame(rows)
    out_single = args.output.strip()
    out_dir = args.output_dir.strip()

    def convert_to_jsonable(v: Any) -> Any:
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception as e:
                raise RuntimeError(f"numpy scalar conversion failed for {type(v).__name__}") from e
        if isinstance(v, dict):
            return {str(k): convert_to_jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert_to_jsonable(x) for x in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    def _write_jsonl_from_df(path: str, frame: "pd.DataFrame") -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as jf:
            for _, row in frame.iterrows():
                obj = {k: convert_to_jsonable(row[k]) for k in frame.columns}
                jf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if out_single:
        parent_out = os.path.dirname(out_single)
        if parent_out:
            os.makedirs(parent_out, exist_ok=True)
        df.to_parquet(out_single, index=False)
        print(f"Wrote {len(df)} rows -> {out_single}")
        if args.write_jsonl:
            base, ext = os.path.splitext(out_single)
            jsonl_path = f"{base}.jsonl" if ext else f"{out_single}.jsonl"
            _write_jsonl_from_df(jsonl_path, df)
            print(f"Wrote {len(df)} rows -> {jsonl_path}")
    else:
        if not out_dir:
            raise SystemExit("Provide --output-dir or --output")
        os.makedirs(out_dir, exist_ok=True)
        vf = float(args.val_fraction)
        if not (0.0 <= vf < 1.0):
            raise SystemExit("val-fraction must be in [0, 1)")
        if vf == 0.0:
            train_df = df
            val_df = df.iloc[0:0].copy()
        else:
            df2 = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
            n_val = max(1, int(len(df2) * vf))
            val_df = df2.iloc[:n_val].copy()
            train_df = df2.iloc[n_val:].copy()
        train_path = os.path.join(out_dir, "train.parquet")
        val_path = os.path.join(out_dir, "val.parquet")
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        print(f"Wrote train {len(train_df)} rows -> {train_path}")
        print(f"Wrote val   {len(val_df)} rows -> {val_path}")
        if args.write_jsonl:
            train_jsonl = os.path.join(out_dir, "train.jsonl")
            val_jsonl = os.path.join(out_dir, "val.jsonl")
            _write_jsonl_from_df(train_jsonl, train_df)
            _write_jsonl_from_df(val_jsonl, val_df)
            print(f"Wrote train {len(train_df)} rows -> {train_jsonl}")
            print(f"Wrote val   {len(val_df)} rows -> {val_jsonl}")

    print(
        "Aggregate stats:",
        {
            "episodes_written": len(rows),
            "episodes_skipped": tot_stats["episodes_skipped"],
            "retrieve_inserted": tot_stats["retrieve_inserted"],
            "llm_queries": tot_stats["llm_queries"],
            "heuristic_queries": tot_stats["heuristic_queries"],
            "assistant_turns": tot_stats["turns"],
            "skipped_gpt": tot_stats["skipped_gpt"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
