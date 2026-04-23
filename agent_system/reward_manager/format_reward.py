# Copyright 2025 MemAdaptor
#
# Format rewards aligned with env projection:
# - Default benches: ``<think>``, ``<action>``, optional ``<memory_retrieve>``.
# - Search bench: ``<think>``, exclusive ``<search>`` xor ``<answer>`` (see ``search_projection``).
# Segment caps (see ``max_*_segments`` in config): AlfWorld projection uses the first ``<action>`` pair only;
# ``MemoryManager.extract_query`` uses the first retrieval block only — rewards should not treat duplicates as valid.
# Multi-turn: ``EpisodeRewardManager`` uses :func:`compute_generic_action_think_memory_format_reward_multi_step`:
# think/action bounds apply **per step**; memory-retrieve count for scoring defaults to summing XML
# segments, but should match the env counter ``memory_retrieval_counts`` when passed in so reward
# aligns with ``memory/retrieval_count_*`` (see ``env_trajectory_memory_retrieval_count``).

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

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
    "format_think_over_limit",
    "format_action_over_limit",
    "format_memory_over_limit",
    "format_protocol_hard_zero",
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


def _bounded_count_ok(
    n: int,
    *,
    at_least: int,
    at_most: int,
) -> tuple[float, bool]:
    """Return (ok 0/1, over_limit). ``at_most <= 0`` disables the upper bound."""
    if at_most <= 0:
        return (1.0 if n >= at_least else 0.0, False)
    if n < at_least:
        return (0.0, False)
    if n > at_most:
        return (0.0, True)
    return (1.0, False)


def _optional_memory_ok(
    mem_n: int,
    *,
    max_segments: int,
) -> tuple[float, bool]:
    """When memory retrieve is optional: 0 blocks ok; if any, require 1..max when max>0."""
    if max_segments <= 0:
        return (1.0, False)
    if mem_n == 0:
        return (1.0, False)
    if mem_n > max_segments:
        return (0.0, True)
    return (1.0, False)


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
    max_think_segments: int = 0,
    max_action_segments: int = 1,
    max_memory_retrieve_segments: int = 1,
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

    think_ok, think_over = _bounded_count_ok(think_n, at_least=1, at_most=max_think_segments)
    action_ok, action_over = _bounded_count_ok(action_n, at_least=1, at_most=max_action_segments)

    if require_memory_retrieve:
        mem_ok, mem_over = _bounded_count_ok(mem_n, at_least=1, at_most=max_memory_retrieve_segments)
        w_mem = weight_memory
    else:
        mem_ok, mem_over = _optional_memory_ok(mem_n, max_segments=max_memory_retrieve_segments)
        w_mem = 0.0

    metrics["format_think_over_limit"] = int(think_over)
    metrics["format_action_over_limit"] = int(action_over)
    metrics["format_memory_over_limit"] = int(mem_over)

    optional_memory_overflow = (
        not require_memory_retrieve
        and w_mem <= 0.0
        and max_memory_retrieve_segments > 0
        and mem_over
    )
    metrics["format_protocol_hard_zero"] = int(optional_memory_overflow)

    if optional_memory_overflow:
        metrics["format_think_ok"] = int(think_ok)
        metrics["format_action_ok"] = int(action_ok)
        metrics["format_memory_ok"] = int(mem_ok)
        return GenericFormatRewardOutput(reward=0.0, metrics=metrics)

    denom = weight_think + weight_action + w_mem
    score = (weight_think * think_ok + weight_action * action_ok + w_mem * mem_ok) / denom
    metrics["format_think_ok"] = int(think_ok)
    metrics["format_action_ok"] = int(action_ok)
    metrics["format_memory_ok"] = int(mem_ok)
    return GenericFormatRewardOutput(reward=float(score), metrics=metrics)


def compute_generic_action_think_memory_format_reward_multi_step(
    step_responses: list[str],
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
    max_think_segments: int = 0,
    max_action_segments: int = 1,
    max_memory_retrieve_segments: int = 1,
    env_trajectory_memory_retrieval_count: Optional[int] = None,
) -> GenericFormatRewardOutput:
    """Format reward over a full trajectory (multi-turn rollout rows share ``traj_uid``).

    - **think / action**: each non-empty step must satisfy the same per-step bounds as
      :func:`compute_generic_action_think_memory_format_reward`.
    - **memory retrieve**: when ``env_trajectory_memory_retrieval_count`` is set, that value
      (same rule as rollout ``memory_retrieval_counts``) is used for the trajectory-wide memory
      count and for ``format_memory_retrieve_count``; otherwise valid XML segments are summed
      across steps and compared to ``max_memory_retrieve_segments``.
    """
    texts = [s or "" for s in step_responses]
    metrics = _zero_metrics()
    if not texts:
        return GenericFormatRewardOutput(reward=0.0, metrics=metrics)

    if penalize_chinese_chars:
        for t in texts:
            if _has_chinese(t):
                metrics["format_has_chinese"] = 1
                metrics["format_penalized_chinese"] = 1
                return GenericFormatRewardOutput(reward=0.0, metrics=metrics)

    think_ok_all = True
    action_ok_all = True
    think_over_any = False
    action_over_any = False
    sum_think = 0
    sum_action = 0
    sum_mem = 0

    for t in texts:
        if not t.strip():
            think_ok_all = False
            action_ok_all = False
            continue
        think_n = _valid_segments(t, think_open_tag, think_close_tag, min_segment_chars)
        action_n = _valid_segments(t, action_open_tag, action_close_tag, min_segment_chars)
        mem_n = _valid_segments(t, memory_open_tag, memory_close_tag, min_segment_chars)
        sum_think += think_n
        sum_action += action_n
        sum_mem += mem_n

        to, tover = _bounded_count_ok(think_n, at_least=1, at_most=max_think_segments)
        ao, aover = _bounded_count_ok(action_n, at_least=1, at_most=max_action_segments)
        if to == 0.0:
            think_ok_all = False
        if ao == 0.0:
            action_ok_all = False
        if tover:
            think_over_any = True
        if aover:
            action_over_any = True

    if env_trajectory_memory_retrieval_count is not None:
        try:
            sum_mem = max(0, int(env_trajectory_memory_retrieval_count))
        except (TypeError, ValueError):
            pass

    metrics["format_think_count"] = sum_think
    metrics["format_action_count"] = sum_action
    metrics["format_memory_retrieve_count"] = sum_mem

    think_ok_f = 1.0 if think_ok_all else 0.0
    action_ok_f = 1.0 if action_ok_all else 0.0

    if require_memory_retrieve:
        mem_ok, mem_over = _bounded_count_ok(
            sum_mem, at_least=1, at_most=max_memory_retrieve_segments
        )
        w_mem = weight_memory
    else:
        mem_ok, mem_over = _optional_memory_ok(sum_mem, max_segments=max_memory_retrieve_segments)
        w_mem = 0.0

    metrics["format_think_over_limit"] = int(think_over_any)
    metrics["format_action_over_limit"] = int(action_over_any)
    metrics["format_memory_over_limit"] = int(mem_over)

    optional_memory_overflow = (
        not require_memory_retrieve
        and w_mem <= 0.0
        and max_memory_retrieve_segments > 0
        and mem_over
    )
    metrics["format_protocol_hard_zero"] = int(optional_memory_overflow)

    if optional_memory_overflow:
        metrics["format_think_ok"] = int(think_ok_f)
        metrics["format_action_ok"] = int(action_ok_f)
        metrics["format_memory_ok"] = int(mem_ok)
        return GenericFormatRewardOutput(reward=0.0, metrics=metrics)

    denom = weight_think + weight_action + w_mem
    score = (weight_think * think_ok_f + weight_action * action_ok_f + w_mem * mem_ok) / denom
    metrics["format_think_ok"] = int(think_ok_f)
    metrics["format_action_ok"] = int(action_ok_f)
    metrics["format_memory_ok"] = int(mem_ok)
    return GenericFormatRewardOutput(reward=float(score), metrics=metrics)


def compute_search_think_memory_format_reward_multi_step(
    step_responses: list[str],
    **kwargs,
) -> GenericFormatRewardOutput:
    """Search-bench format reward: conservative **min** over per-step scores (protocol is step-local)."""
    outs: list[GenericFormatRewardOutput] = []
    for t in step_responses:
        if not (t or "").strip():
            outs.append(GenericFormatRewardOutput(reward=0.0, metrics=_zero_metrics()))
            continue
        outs.append(compute_search_think_memory_format_reward(t, **kwargs))
    if not outs:
        return GenericFormatRewardOutput(reward=0.0, metrics=_zero_metrics())
    worst = min(outs, key=lambda o: o.reward)
    return worst


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
    max_think_segments: int = 0,
    max_memory_retrieve_segments: int = 1,
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

    think_ok, think_over = _bounded_count_ok(think_n, at_least=1, at_most=max_think_segments)
    if require_memory_retrieve:
        mem_ok, mem_over = _bounded_count_ok(mem_n, at_least=1, at_most=max_memory_retrieve_segments)
        w_mem = weight_memory
    else:
        mem_ok, mem_over = _optional_memory_ok(mem_n, max_segments=max_memory_retrieve_segments)
        w_mem = 0.0

    metrics["format_think_over_limit"] = int(think_over)
    metrics["format_action_over_limit"] = 0
    metrics["format_memory_over_limit"] = int(mem_over)

    optional_memory_overflow = (
        not require_memory_retrieve
        and w_mem <= 0.0
        and max_memory_retrieve_segments > 0
        and mem_over
    )
    metrics["format_protocol_hard_zero"] = int(optional_memory_overflow)

    if optional_memory_overflow:
        metrics["format_think_ok"] = int(think_ok)
        metrics["format_action_ok"] = int(protocol_ok)
        metrics["format_memory_ok"] = int(mem_ok)
        return GenericFormatRewardOutput(reward=0.0, metrics=metrics)

    denom = weight_think + weight_protocol + w_mem
    score = (weight_think * think_ok + weight_protocol * protocol_ok + w_mem * mem_ok) / denom
    metrics["format_think_ok"] = int(think_ok)
    metrics["format_action_ok"] = int(protocol_ok)
    metrics["format_memory_ok"] = int(mem_ok)
    return GenericFormatRewardOutput(reward=float(score), metrics=metrics)


def _run_self_tests() -> None:
    T = dict(
        min_segment_chars=3,
        require_memory_retrieve=False,
        memory_open_tag="<memory_retrieve>",
        memory_close_tag="</memory_retrieve>",
        think_open_tag="<think>",
        think_close_tag="</think>",
        action_open_tag="<action>",
        action_close_tag="</action>",
        weight_think=1.0,
        weight_action=1.0,
        weight_memory=1.0,
        penalize_chinese_chars=False,
        max_think_segments=0,
        max_action_segments=1,
        max_memory_retrieve_segments=1,
    )

    def g(text: str, **over) -> GenericFormatRewardOutput:
        kw = {**T, **over}
        return compute_generic_action_think_memory_format_reward(text, **kw)

    # 1) 单步合法：think + action（段内长度 >= min_segment_chars）
    o1 = g("<think>abc</think><action>goto</action>")
    assert o1.reward == 1.0, o1

    # 2) 两个 action：超限，加权后 0.5
    o2 = g(
        "<think>abc</think>"
        "<action>one</action><action>two</action>",
    )
    assert abs(o2.reward - 0.5) < 1e-9 and o2.metrics["format_action_over_limit"] == 1, o2

    # 3) 可选记忆但两段检索：整段格式分为 0
    o3 = g(
        "<think>abc</think><action>run</action>"
        "<memory_retrieve>que1</memory_retrieve><memory_retrieve>que2</memory_retrieve>",
    )
    assert o3.reward == 0.0 and o3.metrics["format_protocol_hard_zero"] == 1, o3

    # 4) 必选记忆 + 单段：满分
    o4 = compute_generic_action_think_memory_format_reward(
        "<think>abc</think><action>run</action>"
        "<memory_retrieve>query</memory_retrieve>",
        **{**T, "require_memory_retrieve": True},
    )
    assert o4.reward == 1.0, o4

    # 5) 必选记忆 + 两段检索：记忆超限
    o5 = compute_generic_action_think_memory_format_reward(
        "<think>abc</think><action>run</action>"
        "<memory_retrieve>que1</memory_retrieve><memory_retrieve>que2</memory_retrieve>",
        **{**T, "require_memory_retrieve": True},
    )
    assert o5.metrics["format_memory_over_limit"] == 1 and o5.reward < 1.0, o5

    # 5b) 上限=2 却出现 3 段有效检索：必选记忆下 mem_ok=0，格式分 = (1+1+0)/3
    o5b = compute_generic_action_think_memory_format_reward(
        "<think>abc</think><action>run</action>"
        "<memory_retrieve>que1</memory_retrieve>"
        "<memory_retrieve>que2</memory_retrieve>"
        "<memory_retrieve>que3</memory_retrieve>",
        **{**T, "require_memory_retrieve": True, "max_memory_retrieve_segments": 2},
    )
    assert o5b.metrics["format_memory_over_limit"] == 1 and o5b.metrics["format_memory_retrieve_count"] == 3, o5b
    assert abs(o5b.reward - 2.0 / 3.0) < 1e-9, o5b

    # 5c) 可选记忆 + 三段检索：超过 max_memory_retrieve_segments=1，整段格式分硬清零
    o5c = g(
        "<think>abc</think><action>run</action>"
        "<memory_retrieve>que1</memory_retrieve>"
        "<memory_retrieve>que2</memory_retrieve>"
        "<memory_retrieve>que3</memory_retrieve>",
    )
    assert o5c.reward == 0.0 and o5c.metrics["format_protocol_hard_zero"] == 1, o5c

    # 6) max_action_segments=0：允许多 action（legacy）
    o6 = g(
        "<think>abc</think>"
        "<action>aaa</action><action>bbb</action>",
        max_action_segments=0,
    )
    assert abs(o6.reward - 1.0) < 1e-9, o6

    # 7) Search 台：think + 单次 search
    S = dict(
        min_segment_chars=3,
        require_memory_retrieve=False,
        memory_open_tag="<memory_retrieve>",
        memory_close_tag="</memory_retrieve>",
        think_open_tag="<think>",
        think_close_tag="</think>",
        search_open_tag="<search>",
        search_close_tag="</search>",
        answer_open_tag="<answer>",
        answer_close_tag="</answer>",
        weight_think=1.0,
        weight_protocol=1.0,
        weight_memory=1.0,
        penalize_chinese_chars=False,
        max_think_segments=0,
        max_memory_retrieve_segments=1,
    )
    s1 = compute_search_think_memory_format_reward(
        "<think>abc</think><search>find it</search>",
        **S,
    )
    assert abs(s1.reward - 1.0) < 1e-9, s1

    # 8) Search：search 与 answer 同时出现 -> protocol 无效
    s2 = compute_search_think_memory_format_reward(
        "<think>abc</think>"
        "<search>foo</search><answer>bar</answer>",
        **S,
    )
    assert s2.metrics["format_action_ok"] == 0 and s2.reward < 1.0, s2

    # 9) 中文惩罚
    o7 = g("<think>abc</think><action>run</action>中文", penalize_chinese_chars=True)
    assert o7.reward == 0.0 and o7.metrics["format_penalized_chinese"] == 1, o7

    # 10) 数据源子串
    assert use_search_format_reward("my_search_task", ["search"]) is True
    assert use_search_format_reward("alfworld", ["search"]) is False

    # 11) 轨迹：两步各 1 次检索，max_memory_retrieve_segments=1（可选记忆）→ 总次数 2 硬清零
    m11 = compute_generic_action_think_memory_format_reward_multi_step(
        [
            "<think>abc</think><action>run</action><memory_retrieve>que1</memory_retrieve>",
            "<think>def</think><action>go</action><memory_retrieve>que2</memory_retrieve>",
        ],
        **T,
    )
    assert m11.reward == 0.0 and m11.metrics["format_protocol_hard_zero"] == 1
    assert m11.metrics["format_memory_retrieve_count"] == 2, m11

    # 12) 轨迹：两步共 1 次检索，上限 1 → 满分
    m12 = compute_generic_action_think_memory_format_reward_multi_step(
        [
            "<think>abc</think><action>run</action>",
            "<think>def</think><action>goto</action><memory_retrieve>que1</memory_retrieve>",
        ],
        **T,
    )
    assert abs(m12.reward - 1.0) < 1e-9 and m12.metrics["format_memory_retrieve_count"] == 1, m12

    # 13) 与 env 计数对齐：无 XML 检索段，但 rollout 记 2 次检索 → 计入 format_memory_retrieve_count，可选记忆下超限硬清零
    m13 = compute_generic_action_think_memory_format_reward_multi_step(
        [
            "<think>abc</think><action>run</action>",
            "<think>def</think><action>go</action>",
        ],
        **T,
        env_trajectory_memory_retrieval_count=2,
    )
    assert (
        m13.metrics["format_memory_retrieve_count"] == 2
        and m13.metrics["format_memory_over_limit"] == 1
        and m13.metrics["format_protocol_hard_zero"] == 1
        and m13.reward == 0.0
    ), m13

    print("format_reward self-tests: OK (15 checks)")


if __name__ == "__main__":
    _run_self_tests()
