# Copyright 2025 MemAdaptor
#
# Format rewards aligned with env projection:
# - Default benches: ``<think>``, ``<action>``, optional ``<memory_retrieve>``.
# - Search bench: ``<think>``, exclusive ``<search>`` xor ``<answer>`` (see ``search_projection``).

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Keys appended per row to ``reward_extra_info`` (aligned lengths for trainer logging).
FORMAT_REWARD_EXTRA_KEYS = (
    "format_think_count",
    "format_action_count",
    "format_memory_retrieve_count",
    "format_has_chinese",
    "format_penalized_chinese",
    "format_think_ok",
    "format_action_ok",
    "format_memory_ok",
)


@dataclass
class GenericFormatRewardOutput:
    reward: float
    metrics: dict[str, Any]


def _zero_metrics() -> dict[str, Any]:
    return {k: 0 for k in FORMAT_REWARD_EXTRA_KEYS}


def empty_format_reward_metrics() -> dict[str, Any]:
    return _zero_metrics()


def use_search_format_reward(data_source: str, search_data_source_substrings: list[str]) -> bool:
    ds = data_source.lower()
    for sub in search_data_source_substrings:
        if sub.lower() in ds:
            return True
    return False


def _valid_segments(text: str, open_tag: str, close_tag: str, min_chars: int) -> int:
    pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
    blocks = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return len([b for b in blocks if len(b.strip()) >= min_chars])


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def compute_generic_action_think_memory_format_reward(
    response_str: str,
    *,
    min_segment_chars: int,
    require_memory_retrieve: bool,
    memory_open_tag: str,
    memory_close_tag: str,
    think_open_tag: str,
    think_close_tag: str,
    action_open_tag: str,
    action_close_tag: str,
    weight_think: float,
    weight_action: float,
    weight_memory: float,
    penalize_chinese_chars: bool,
) -> GenericFormatRewardOutput:
    text = response_str or ""
    metrics = _zero_metrics()

    if penalize_chinese_chars and _has_chinese(text):
        metrics["format_has_chinese"] = 1
        metrics["format_penalized_chinese"] = 1
        return GenericFormatRewardOutput(reward=0.0, metrics=metrics)

    think_n = _valid_segments(text, think_open_tag, think_close_tag, min_segment_chars)
    action_n = _valid_segments(text, action_open_tag, action_close_tag, min_segment_chars)
    mem_n = _valid_segments(text, memory_open_tag, memory_close_tag, min_segment_chars)

    metrics["format_think_count"] = think_n
    metrics["format_action_count"] = action_n
    metrics["format_memory_retrieve_count"] = mem_n

    think_ok = 1.0 if think_n > 0 else 0.0
    action_ok = 1.0 if action_n > 0 else 0.0
    if require_memory_retrieve:
        mem_ok = 1.0 if mem_n > 0 else 0.0
        w_mem = weight_memory
    else:
        mem_ok = 1.0
        w_mem = 0.0

    denom = weight_think + weight_action + w_mem
    score = (weight_think * think_ok + weight_action * action_ok + w_mem * mem_ok) / denom
    metrics["format_think_ok"] = int(think_ok)
    metrics["format_action_ok"] = int(action_ok)
    metrics["format_memory_ok"] = int(mem_ok)
    return GenericFormatRewardOutput(reward=float(score), metrics=metrics)


def compute_search_think_memory_format_reward(
    response_str: str,
    *,
    min_segment_chars: int,
    require_memory_retrieve: bool,
    memory_open_tag: str,
    memory_close_tag: str,
    think_open_tag: str,
    think_close_tag: str,
    search_open_tag: str,
    search_close_tag: str,
    answer_open_tag: str,
    answer_close_tag: str,
    weight_think: float,
    weight_protocol: float,
    weight_memory: float,
    penalize_chinese_chars: bool,
) -> GenericFormatRewardOutput:
    """Match ``search_projection`` validity: one of search xor answer, non-empty body, no duplicate / mixed tags."""
    text = response_str or ""
    metrics = _zero_metrics()

    if penalize_chinese_chars and _has_chinese(text):
        metrics["format_has_chinese"] = 1
        metrics["format_penalized_chinese"] = 1
        return GenericFormatRewardOutput(reward=0.0, metrics=metrics)

    re_search_block = re.compile(
        re.escape(search_open_tag) + r"(.*?)" + re.escape(search_close_tag),
        re.IGNORECASE | re.DOTALL,
    )
    re_answer_block = re.compile(
        re.escape(answer_open_tag) + r"(.*?)" + re.escape(answer_close_tag),
        re.IGNORECASE | re.DOTALL,
    )
    re_search_tag = re.compile(re.escape(search_open_tag), re.IGNORECASE)
    re_answer_tag = re.compile(re.escape(answer_open_tag), re.IGNORECASE)

    n_search = len(re_search_tag.findall(text))
    n_answer = len(re_answer_tag.findall(text))

    protocol_ok = 0.0
    body_nonempty = False
    if n_search and n_answer:
        protocol_ok = 0.0
    elif n_search > 1 or n_answer > 1:
        protocol_ok = 0.0
    else:
        m = re_search_block.search(text)
        if m:
            body_nonempty = len(m.group(1).strip()) >= min_segment_chars
        else:
            m = re_answer_block.search(text)
            if m:
                body_nonempty = len(m.group(1).strip()) >= min_segment_chars
        if body_nonempty and ((n_search == 1 and n_answer == 0) or (n_answer == 1 and n_search == 0)):
            protocol_ok = 1.0

    think_n = _valid_segments(text, think_open_tag, think_close_tag, min_segment_chars)
    mem_n = _valid_segments(text, memory_open_tag, memory_close_tag, min_segment_chars)

    metrics["format_think_count"] = think_n
    metrics["format_action_count"] = int(protocol_ok)
    metrics["format_memory_retrieve_count"] = mem_n

    think_ok = 1.0 if think_n > 0 else 0.0
    if require_memory_retrieve:
        mem_ok = 1.0 if mem_n > 0 else 0.0
        w_mem = weight_memory
    else:
        mem_ok = 1.0
        w_mem = 0.0

    denom = weight_think + weight_protocol + w_mem
    score = (weight_think * think_ok + weight_protocol * protocol_ok + w_mem * mem_ok) / denom
    metrics["format_think_ok"] = int(think_ok)
    metrics["format_action_ok"] = int(protocol_ok)
    metrics["format_memory_ok"] = int(mem_ok)
    return GenericFormatRewardOutput(reward=float(score), metrics=metrics)
