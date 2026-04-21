# Copyright 2025 MemAdaptor
#
# Online experience / memory write-back: LLM extracts JSON ``{"memories":[...]}`` per trajectory, then VDB insert.
# - mode=self: batch prompts to ``actor_rollout_wg.generate_sequences`` (same Reasoning rollout worker).
# - mode=teacher: OpenAI-compatible ``/v1/chat/completions`` on the driver (no Ray worker).
# - schema=compact: small models output only ``memory_text`` (+ optional ``source_step``); ``state_text``/``action_text``
#   are filled from the trajectory (EvolveR-style principles). schema=full matches ``extract_memory_records.py``.

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import urllib.error
import urllib.request
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer

from agent_system.memory.experience_utility import UTILITY_SCORE_KEY, initial_utility_metadata
from agent_system.memory.memory_manager import MemoryManager, normalize_text
from agent_system.memory.types import MemoryRecord
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)

# Defaults aligned with ``scripts/extract_memory_records.py`` (JSON extraction schema).
DEFAULT_JSON_SYSTEM_PROMPT = """You are an expert agent-memory extraction system.

Your job is to read one successful trajectory from an interactive agent benchmark and extract a small set of reusable memories.

The extracted memories must satisfy all requirements:
1. Each memory should correspond to one local decision point or one short reusable principle grounded in the trajectory.
2. A memory must be useful for retrieval in a future similar state.
3. Do not summarize the whole trajectory into one global high-level paragraph.
4. Prefer step-level or subgoal-level memories.
5. Do not invent facts that are not supported by the trajectory.
6. The output must be valid JSON only, with no markdown fences and no extra explanation.
"""

DEFAULT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE = """Extract at most {num_memories} reusable memories from the following successful trajectory.

Task requirements:
- The benchmark is "{task_name}".
- The trajectory is successful, so the extracted memories should be treated as positive memories.
- Each memory should be suitable for naive retrieval later.
- Each memory should contain:
  - source_step: the step index in the trajectory where this memory is grounded. Use 1-based indexing over agent action turns.
  - state_text: a concise but sufficient description of the state/observation that would be useful as the retrieval key.
  - action_text: the actual helpful action from the trajectory, or the directly grounded action-oriented response.
  - memory_text: a short reusable memory written for future prompting. It should emphasize what to do in a similar local situation, and may mention important preconditions.
  - value: an optional initial quality/value estimate in [0, 1]. Use null if you are unsure.
  - metadata: an object that may include short fields such as "subgoal", "preconditions", "why_useful".

Important extraction rules:
- Keep memories local and concrete.
- Do not output duplicate memories.
- Do not output more than {num_memories} memories.
- If the trajectory is short, output fewer memories.
- Preserve benchmark-specific details when they are critical (e.g., object-state relations, environment preconditions, product constraints, or option-selection cues depending on the current task).

Return JSON with exactly this schema:
{{
  "memories": [
    {{
      "source_step": 1,
      "state_text": "...",
      "action_text": "...",
      "memory_text": "...",
      "metadata": {{
        "subgoal": "...",
        "preconditions": "...",
        "why_useful": "..."
      }}
    }}
  ]
}}

Trajectory:
{trajectory_text}
"""

# Fewer fields for weak extractors (small models / self-distill). Model outputs ``memory_text`` (+ optional ``source_step``);
# ``state_text`` / ``action_text`` are filled from the trajectory when possible (EvolveR-style “Guiding Principle” in one line).
DEFAULT_COMPACT_JSON_SYSTEM_PROMPT = """You analyze agent trajectories and distill generalizable wisdom (Guiding or Warning Principles).
Each principle must be useful later: another run in a similar situation should benefit from retrieving it.

Output one JSON object only. No markdown fences, no commentary before or after the JSON.

Requirements:
- Ground every principle in the trajectory; do not invent facts.
- One clear English sentence per principle (core advice). No bullet lists inside memory_text.
- Prefer strategies that explain what to do or check, not a play-by-play recap of this episode."""

DEFAULT_COMPACT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE = """Benchmark: "{task_name}".
The trajectory below is from a episode of the benchmark. Extract at most {num_memories} Guiding or Warning Principles.
Each principle must be independently useful if stored in a memory bank and retrieved later.

Return JSON exactly in this shape. ``source_step`` is optional (1-based index of the agent turn the principle is grounded in).
{{
  "memories": [
    {{ "memory_text": "one-sentence principle here", "source_step": 1 }}
  ]
}}

Example (different task—format only):
{{
  "memories": [
    {{
      "memory_text": "When a download fails with a 404, verify the URL before retrying instead of repeating the same request.",
      "source_step": 2
    }}
  ]
}}

Trajectory:
{trajectory_text}
"""


def _extraction_schema(es: DictConfig) -> str:
    s = str(es.get("schema", "full") or "full").strip().lower()
    return "compact" if s == "compact" else "full"


def experience_summarizer_active(config: DictConfig) -> bool:
    mem = config.env.memory
    if not bool(mem.enabled):
        return False
    if not bool(mem.write_back):
        return False
    mode = str(mem.experience_summarizer.mode).strip().lower()
    return mode in ("self", "teacher")


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort JSON object parse; matches ``scripts/extract_memory_records.extract_json_object``."""
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


def _action_text_from_step(step: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> str:
    art = step.get("api_response_text")
    if isinstance(art, str) and art.strip():
        return art.strip()
    resp = step.get("responses")
    if isinstance(resp, torch.Tensor):
        return tokenizer.decode(resp.detach().cpu(), skip_special_tokens=True).strip()
    return ""


def _state_text_from_step(step: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> str:
    pr = step.get("prompts")
    if isinstance(pr, torch.Tensor):
        return tokenizer.decode(pr.detach().cpu(), skip_special_tokens=True).strip()
    iid = step.get("input_ids")
    if isinstance(iid, torch.Tensor):
        return tokenizer.decode(iid.detach().cpu(), skip_special_tokens=True).strip()
    return ""


def _format_trajectory_plain(
    steps: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
) -> str:
    lines: List[str] = []
    for t, step in enumerate(steps):
        if not step.get("active_masks", True):
            continue
        act = _action_text_from_step(step, tokenizer)
        st = _state_text_from_step(step, tokenizer)
        lines.append(f"--- Step {t + 1} ---")
        if st:
            lines.append(f"Context (prompt):\n{st}")
        if act:
            lines.append(f"Action (model output):\n{act}")
        rew = step.get("rewards")
        if rew is not None:
            lines.append(f"Step reward: {rew}")
    body = "\n\n".join(lines)
    return body


def _step_outputs_memory_retrieve(step: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> bool:
    return "<memory_retrieve>" in _action_text_from_step(step, tokenizer).lower()


def _ordered_active_steps(steps: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    return [(t, steps[t]) for t in range(len(steps)) if steps[t].get("active_masks", True)]


def _select_ordered_steps_head_tail_protected(
    ordered: List[Tuple[int, Dict[str, Any]]],
    tokenizer: PreTrainedTokenizer,
    episode_max_turns: int,
    head_turns: int,
    tail_turns: int,
) -> tuple[List[Tuple[int, Dict[str, Any]]], dict[str, Any]]:
    """Keep first/last turns when long; keep middle turns that touch memory retrieval (same idea as ``mem_fail_judge``)."""
    n = len(ordered)
    meta: dict[str, Any] = {
        "total_turns": n,
        "turns_included": n,
        "omitted_middle_turns": 0,
        "head_turns_kept": n,
        "tail_turns_kept": n,
        "truncated_by_turns": False,
        "protected_memory_retrieve_positions": [],
    }
    if episode_max_turns <= 0 or n <= episode_max_turns:
        return ordered, meta

    h = max(0, head_turns)
    t = max(0, tail_turns)
    if h + t >= n:
        return ordered, meta

    keep_pos: set[int] = set(range(h)) | set(range(n - t, n))
    protected_positions: list[int] = []
    for i in range(h, n - t):
        _, st = ordered[i]
        if _step_outputs_memory_retrieve(st, tokenizer):
            keep_pos.add(i)
            protected_positions.append(i)
    for i in range(n - 1):
        if _step_outputs_memory_retrieve(ordered[i][1], tokenizer):
            keep_pos.add(i + 1)

    selected = [ordered[i] for i in sorted(keep_pos)]
    omitted = n - len(selected)
    meta["truncated_by_turns"] = omitted > 0
    meta["turns_included"] = len(selected)
    meta["omitted_middle_turns"] = omitted
    meta["head_turns_kept"] = h
    meta["tail_turns_kept"] = t
    meta["protected_memory_retrieve_positions"] = protected_positions
    return selected, meta


def _format_ordered_steps_plain(
    ordered: List[Tuple[int, Dict[str, Any]]],
    tokenizer: PreTrainedTokenizer,
    *,
    coverage_preamble: str = "",
) -> str:
    lines: List[str] = []
    if coverage_preamble.strip():
        lines.append(coverage_preamble.strip())
    for orig_i, step in ordered:
        act = _action_text_from_step(step, tokenizer)
        st = _state_text_from_step(step, tokenizer)
        lines.append(f"--- Step {orig_i + 1} ---")
        if st:
            lines.append(f"Context (prompt):\n{st}")
        if act:
            lines.append(f"Action (model output):\n{act}")
        rew = step.get("rewards")
        if rew is not None:
            lines.append(f"Step reward: {rew}")
    body = "\n\n".join(lines)
    return body


def _protection_mask_ordered(ordered: List[Tuple[int, Dict[str, Any]]], tokenizer: PreTrainedTokenizer) -> List[bool]:
    """Steps to avoid dropping when shrinking the trajectory for the summarizer prompt budget.

    Only **model actions** that emit a memory-retrieve query are protected, plus the **next** step (injection
    / continuation). Do **not** key off the full observation prompt: task instructions often contain the
    literal substring ``<memory_retrieve>``, which would mark every turn protected and make token trim fail
    with "all turns are retrieve-protected".
    """
    prot = [False] * len(ordered)
    for i, (_, st) in enumerate(ordered):
        if _step_outputs_memory_retrieve(st, tokenizer):
            prot[i] = True
    for i in range(len(ordered) - 1):
        if _step_outputs_memory_retrieve(ordered[i][1], tokenizer):
            prot[i + 1] = True
    return prot


def _count_summarizer_chat_tokens(
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    user_message: str,
    apply_kw: Optional[dict],
) -> int:
    chat = _build_chat_prompt(tokenizer, system_prompt, user_message, apply_kw if isinstance(apply_kw, dict) else None)
    return len(tokenizer.encode(chat, add_special_tokens=False))


def _coverage_note_from_meta(meta: dict[str, Any], es: DictConfig) -> str:
    if not bool(es.summarizer_trajectory_include_coverage_note):
        return ""
    if not meta["truncated_by_turns"]:
        return ""
    extra = ""
    prot = meta["protected_memory_retrieve_positions"]
    if prot:
        extra = (
            "Middle steps containing <memory_retrieve> are kept; protected ordered indices: "
            f"{prot}. "
        )
    return (
        "[Trajectory coverage] "
        f"Total active steps: {meta['total_turns']}. "
        f"Shown: first {meta['head_turns_kept']} and last {meta['tail_turns_kept']} steps plus protected retrieve-related steps; "
        f"omitted from middle: {meta['omitted_middle_turns']}. "
        f"{extra}\n\n"
    )


def _trim_ordered_steps_to_prompt_token_budget(
    ordered: List[Tuple[int, Dict[str, Any]]],
    *,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    trajectory_user_prompt_template: str,
    template_format_kwargs: Dict[str, Any],
    apply_kw: Optional[dict],
    max_prompt_tokens: int,
    min_turns: int,
    coverage_preamble: str,
) -> Optional[str]:
    """
    Drop removable turns until the full chat fits ``max_prompt_tokens``.

    **Alternating peel on the removable index set** (``rem`` = indices where ``prot[i]`` is false):

    Example: after head/tail compression, 6 steps remain at indices ``0..5`` in time order.
    Suppose only ``1`` and ``4`` are removable (others are retrieve-protected). Then
    ``rem == [1, 4]``, ``max(rem)==4``, ``min(rem)==1``.

    - Round 1 (``peel_right=True``): delete **rightmost** removable → drop index **4**.
    - Round 2 (``peel_right=False``): delete **leftmost** removable among what remains → drop index **1**
      (same physical turn “slot” as before renumbering; list shortens so indices shift).

    Recompute ``rem`` each iteration. If the next removable set were ``[0,2,3,5]``, the same rule applies:
    alternate **``max(rem)``** and **``min(rem)``**. Intuition: peel one layer from the **late** side, then one
    from the **early** side, so you do not delete only the end or only the beginning.
    """
    cur: List[Tuple[int, Dict[str, Any]]] = list(ordered)
    if len(cur) < min_turns:
        return None
    peel_right = True
    # At most one turn removed per iteration until len(cur)==min_turns or prompt fits; +1 for final "fits" check; +8 slack.
    max_iters = max(len(cur) - min_turns + 1, 1) + 8
    for _ in range(max_iters):
        prot = _protection_mask_ordered(cur, tokenizer)
        plain = _format_ordered_steps_plain(cur, tokenizer, coverage_preamble=coverage_preamble)
        if not plain.strip():
            return None
        try:
            user_msg = trajectory_user_prompt_template.format(**{**template_format_kwargs, "trajectory_text": plain})
        except KeyError as e:
            logger.warning("experience_summarizer: template format failed during trim: %s", e)
            return None
        ntok = _count_summarizer_chat_tokens(tokenizer, system_prompt, user_msg, apply_kw)
        if ntok <= max_prompt_tokens:
            return plain
        rem = [i for i in range(len(cur)) if not prot[i]]
        if not rem:
            logger.warning(
                "experience_summarizer: chat still ~%s tokens (> budget %s) but all turns are retrieve-protected; skip.",
                ntok,
                max_prompt_tokens,
            )
            return None
        if len(cur) <= min_turns:
            logger.warning(
                "experience_summarizer: chat ~%s tokens exceeds budget %s at min_turns=%s; skip trajectory.",
                ntok,
                max_prompt_tokens,
                min_turns,
            )
            return None
        drop_i = max(rem) if peel_right else min(rem)
        peel_right = not peel_right
        del cur[drop_i]

    logger.warning("experience_summarizer: token trim exceeded iteration safety; skip.")
    return None


def _prepare_trajectory_text_for_summarizer(
    steps: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    config: DictConfig,
    es: DictConfig,
    *,
    trajectory_user_prompt_template: str,
    template_format_kwargs: Dict[str, Any],
    system_prompt: str,
    apply_kw: Optional[dict],
) -> Optional[str]:
    """
    Build trajectory plaintext for summarizer: optional head/tail + protected retrieve, then optional turn-level
    trimming to satisfy summarizer token budget (full chat = system + user template).
    """
    if not bool(es.summarizer_trajectory_structural_compress):
        plain = _format_trajectory_plain(steps, tokenizer)
        return plain if plain.strip() else None

    ordered = _ordered_active_steps(steps)
    if not ordered:
        return None

    ep_max = int(es.summarizer_trajectory_episode_max_turns)
    head_n = int(es.summarizer_trajectory_head_turns)
    tail_n = int(es.summarizer_trajectory_tail_turns)
    selected, meta = _select_ordered_steps_head_tail_protected(ordered, tokenizer, ep_max, head_n, tail_n)

    min_turns = max(1, int(es.summarizer_trajectory_min_turns_kept))
    if len(selected) < min_turns:
        logger.warning(
            "experience_summarizer: after head/tail selection only %s turns (< min_turns=%s); skip.",
            len(selected),
            min_turns,
        )
        return None

    coverage = _coverage_note_from_meta(meta, es)
    budget = _summarizer_max_prompt_tokens(config)

    if bool(es.summarizer_trajectory_token_trim):
        return _trim_ordered_steps_to_prompt_token_budget(
            selected,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            trajectory_user_prompt_template=trajectory_user_prompt_template,
            template_format_kwargs=template_format_kwargs,
            apply_kw=apply_kw,
            max_prompt_tokens=budget,
            min_turns=min_turns,
            coverage_preamble=coverage,
        )

    plain = _format_ordered_steps_plain(selected, tokenizer, coverage_preamble=coverage)
    return plain if plain.strip() else None


def _episode_success(success: Dict[str, np.ndarray], index: int) -> bool:
    if not success:
        return False
    for _k, arr in success.items():
        a = np.asarray(arr)
        if a.size <= index:
            continue
        try:
            return bool(np.asarray(a[index]).reshape(-1)[0] != 0)
        except Exception:
            return bool(a[index])
    return False


def _dataset_item_id_from_infos(total_infos: List[List[Dict[str, Any]]], env_index: int) -> Any:
    if env_index >= len(total_infos) or not total_infos[env_index]:
        return env_index
    last = total_infos[env_index][-1]
    if not isinstance(last, dict):
        return env_index
    for key in ("item_id", "dataset_item_id", "gamefile", "task_id"):
        if key in last and last[key] is not None:
            return last[key]
    return env_index


def _resolve_trajectory_user_prompt_template(es: DictConfig, schema: str) -> str:
    t = es.trajectory_user_prompt_template
    if str(t).strip():
        return str(t)
    if schema == "compact":
        return DEFAULT_COMPACT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE
    return DEFAULT_JSON_TRAJECTORY_USER_PROMPT_TEMPLATE


def _resolve_system_prompt(es: DictConfig, schema: str) -> str:
    s = str(es.system_prompt).strip()
    if s:
        return s
    if schema == "compact":
        return DEFAULT_COMPACT_JSON_SYSTEM_PROMPT
    return DEFAULT_JSON_SYSTEM_PROMPT


def _build_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    user_content: str,
    apply_chat_template_kwargs: Optional[dict],
) -> str:
    apply_kw = dict(apply_chat_template_kwargs or {})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_kw,
    )


def _row_from_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_prompt_length: int,
    pad_token_id: int,
    truncation: str,
) -> dict:
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
        prompt=prompt,
        tokenizer=tokenizer,
        max_length=max_prompt_length,
        pad_token_id=pad_token_id,
        left_pad=True,
        truncation=truncation,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "position_ids": position_ids[0],
    }


def _normalize_chat_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base + "/chat/completions"


def _http_chat_completion(url: str, headers: Dict[str, str], payload: dict, timeout: float) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"Chat API HTTP {e.code}: {err}") from e
    if not isinstance(body, dict) or "choices" not in body:
        return ""
    choices = body["choices"]
    if not isinstance(choices, list) or not choices:
        return ""
    ch0 = choices[0]
    if not isinstance(ch0, dict) or "message" not in ch0:
        return ""
    msg = ch0["message"]
    if not isinstance(msg, dict) or "content" not in msg or msg["content"] is None:
        return ""
    return str(msg["content"]).strip()


def _summarize_batch_teacher(
    *,
    trajectory_user_messages: List[str],
    system_prompt: str,
    oa: DictConfig,
) -> List[str]:
    """Summarize trajectories using the teacher HTTP API."""

    base_url = str(oa.base_url)
    url = _normalize_chat_url(base_url)
    api_key = str(oa.api_key)
    if not api_key:
        raise ValueError(
            "env.memory.experience_summarizer.openai_api.api_key is empty; set it or OPENAI_API_KEY."
        )
    model = str(oa.model)
    timeout = float(oa.timeout_sec)
    max_tokens = int(oa.max_tokens)
    temperature = float(oa.temperature)
    max_concurrent = int(oa.max_concurrent)
    max_retries = int(oa.max_retries)
    retry_backoff = float(oa.retry_backoff_sec)
    use_json_response = bool(oa.response_format_json)
    extra_headers: Dict[str, str] = dict(OmegaConf.to_container(oa.extra_headers, resolve=True) or {})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        **extra_headers,
    }

    def one(i: int, user_text: str) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": max_tokens,
        }
        if temperature > 0:
            payload["temperature"] = temperature
        else:
            payload["temperature"] = 0
        if use_json_response:
            payload["response_format"] = {"type": "json_object"}

        delay = retry_backoff
        last_err: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                return _http_chat_completion(url, headers, payload, timeout)
            except Exception as e:
                last_err = e
                logger.warning("experience_summarizer teacher attempt %s failed: %s", attempt, e)
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2
        assert last_err is not None
        raise last_err

    if max_concurrent <= 1:
        return [one(i, t) for i, t in enumerate(trajectory_user_messages)]

    out: List[Optional[str]] = [None] * len(trajectory_user_messages)
    with ThreadPoolExecutor(max_workers=min(max_concurrent, len(trajectory_user_messages))) as ex:
        futs = {ex.submit(one, i, trajectory_user_messages[i]): i for i in range(len(trajectory_user_messages))}
        for fut in as_completed(futs):
            idx = futs[fut]
            out[idx] = fut.result()
    return [x or "" for x in out]


def _vllm_rollout_max_prompt_tokens(config: DictConfig) -> int:
    """
    Max prompt tokens that fit in the Reasoning actor's vLLM engine (``max_model_len - response_length``,
    same rule as vllm_rollout_spmd). Self-mode summarizer tokenization uses this cap; turn-level trim
    targets the same budget so the full chat usually fits without post-hoc truncation.
    """
    ro = OmegaConf.select(config, "actor_rollout_ref.rollout")
    if ro is None:
        return max(1, int(config.data.max_prompt_length))
    mml_cfg = ro.max_model_len
    if mml_cfg is None:
        mml = int(ro.prompt_length) + int(ro.response_length)
    else:
        mml = int(mml_cfg)
    roll_resp = int(ro.response_length)
    return max(1, mml - roll_resp)


def _summarizer_max_prompt_tokens(config: DictConfig) -> int:
    """
    Token budget for experience summarization only (trajectory trim + self-mode prompt encoding).

    Multi-turn rollout still uses ``config.data.max_prompt_length``. To summarize with a **larger**
    prompt than rollout, set ``actor_rollout_ref.rollout.max_model_len`` to at least
    ``summarizer_max_prompt_tokens + rollout.response_length``, and set
    ``env.memory.experience_summarizer.summarizer_max_prompt_tokens`` to the desired trim target
    (clamped to ``max_model_len - response_length``).

    When ``summarizer_max_prompt_tokens`` is null or non-positive, uses the full engine cap from
    :func:`_vllm_rollout_max_prompt_tokens`.
    """
    engine_cap = _vllm_rollout_max_prompt_tokens(config)
    es = OmegaConf.select(config, "env.memory.experience_summarizer")
    if es is None:
        return engine_cap
    raw = OmegaConf.select(es, "summarizer_max_prompt_tokens", default=None)
    if raw is None:
        return engine_cap
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return engine_cap
    if v <= 0:
        return engine_cap
    if v > engine_cap:
        logger.warning(
            "experience_summarizer: summarizer_max_prompt_tokens=%s exceeds vLLM prompt cap %s "
            "(raise actor_rollout_ref.rollout.max_model_len); clamping.",
            v,
            engine_cap,
        )
    return min(v, engine_cap)


def _summarize_batch_self(
    *,
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    config: DictConfig,
    es: DictConfig,
    actor_rollout_wg,
) -> List[str]:
    """Summarize trajectories using the actor rollout worker."""
    max_prompt_length = _summarizer_max_prompt_tokens(config)
    truncation = "right"
    ro = OmegaConf.select(config, "actor_rollout_ref.rollout")
    response_length = int(ro.response_length) if ro is not None else int(config.data.max_response_length)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    rows = [_row_from_prompt(tokenizer, p, max_prompt_length, int(pad_id), truncation) for p in prompts]
    data = collate_fn(rows)
    meta = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": int(pad_id),
        "recompute_log_prob": False,
        "do_sample": bool(es.do_sample),
        "validate": False,
        "response_length": response_length,
        "temperature": float(es.temperature),
        "top_p": float(es.top_p),
    }
    dp = DataProto.from_single_dict(data=data, meta_info=meta)
    pad_size = 0
    try:
        world_size = int(actor_rollout_wg.world_size)
    except Exception:
        world_size = 1
    dp_pad, pad_size = pad_dataproto_to_divisor(dp, world_size)
    out_pad = actor_rollout_wg.generate_sequences(dp_pad)
    out = unpad_dataproto(out_pad, pad_size=pad_size)
    n = len(prompts)
    # openai_api rollout: real text is in non_tensor_batch; ``responses`` are placeholders (often pad-only).
    api_txt = out.non_tensor_batch.get("api_response_text") if out.non_tensor_batch else None
    if api_txt is not None:
        texts = []
        for i in range(n):
            if i < len(api_txt):
                cell = api_txt[i]
                texts.append(str(cell).strip() if cell is not None else "")
            else:
                texts.append("")
    else:
        texts = tokenizer.batch_decode(out.batch["responses"][:n], skip_special_tokens=True)
        texts = [t.strip() for t in texts]
    return texts


def _chunk_ranges(length: int, chunk_size: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= length:
        return [(0, length)]
    return [(i, min(i + chunk_size, length)) for i in range(0, length, chunk_size)]


def _parse_memories_from_model_output(raw: str) -> list[dict[str, Any]]:
    stripped = (raw or "").strip()
    if not stripped:
        logger.debug("experience_summarizer: empty extractor output; skip JSON parse")
        return []
    try:
        obj = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as e:
        preview = stripped[:240].replace("\n", " ")
        logger.warning(
            "experience_summarizer: JSON parse failed: %s | preview=%r",
            e,
            preview + ("…" if len(stripped) > 240 else ""),
        )
        return []
    if "memories" not in obj:
        return []
    memories = obj["memories"]
    if not isinstance(memories, list):
        return []
    return [m for m in memories if isinstance(m, dict)]


def _memory_records_from_extractor_output(
    raw: str,
    *,
    pending: Tuple[int, str, float, bool, str, int, Any],
    task_name: str,
    mode: str,
    model_label: str,
    max_pairs: int,
    max_state_chars: int,
    max_action_chars: int,
    max_memory_chars: int,
    schema: str = "full",
    steps: Optional[List[Dict[str, Any]]] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> list[MemoryRecord]:
    """Parse one trajectory's model output and return records that pass field validation."""
    env_index, traj_s, r, succ, traj_plain_for_llm, traj_chars_full, dataset_item_id = pending
    memories = _parse_memories_from_model_output(raw)[:max_pairs]
    out: list[MemoryRecord] = []
    for midx, mem_obj in enumerate(memories):
        rec = _memory_record_from_extracted_dict(
            memory=mem_obj,
            memory_idx=midx,
            task_name=task_name,
            trajectory_index=env_index,
            dataset_item_id=dataset_item_id,
            source_episode_id=traj_s,
            episode_reward=r,
            episode_success=succ,
            mode=mode,
            trajectory_chars_for_prompt=len(traj_plain_for_llm),
            trajectory_chars_full=traj_chars_full,
            extractor_model_label=model_label,
            max_state_chars=max_state_chars,
            max_action_chars=max_action_chars,
            max_memory_chars=max_memory_chars,
            schema=schema,
            steps=steps,
            tokenizer=tokenizer,
        )
        if rec is not None:
            out.append(rec)
    return out


def _run_extraction_inference_for_indices(
    indices: List[int],
    *,
    mode: str,
    trajectory_user_messages: List[str],
    chat_prompts: List[str] | None,
    summarization_batch_size: int,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str,
    config: DictConfig,
    es: DictConfig,
    actor_rollout_wg,
    oa: DictConfig | None,
) -> List[str]:
    """Run teacher/self extraction for ``indices`` (order preserved); returns one raw string per index."""
    if not indices:
        return []
    if mode == "self":
        if chat_prompts is None:
            raise ValueError("chat_prompts required for mode=self")
        sub_prompts = [chat_prompts[i] for i in indices]
        texts: List[str] = []
        for start, end in _chunk_ranges(len(sub_prompts), summarization_batch_size or len(sub_prompts)):
            texts.extend(
                _summarize_batch_self(
                    prompts=sub_prompts[start:end],
                    tokenizer=tokenizer,
                    config=config,
                    es=es,
                    actor_rollout_wg=actor_rollout_wg,
                )
            )
        return texts
    if mode == "teacher":
        if oa is None:
            raise ValueError("openai_api config required for mode=teacher")
        sub_msgs = [trajectory_user_messages[i] for i in indices]
        texts = []
        for start, end in _chunk_ranges(len(sub_msgs), summarization_batch_size or len(sub_msgs)):
            texts.extend(
                _summarize_batch_teacher(
                    trajectory_user_messages=sub_msgs[start:end],
                    system_prompt=system_prompt,
                    oa=oa,
                )
            )
        return texts
    raise ValueError(f"unsupported mode={mode}")


def _raw_trajectory_fallback_memory_record(
    *,
    task_name: str,
    trajectory_index: int,
    dataset_item_id: Any,
    source_episode_id: str,
    episode_reward: float,
    episode_success: bool,
    traj_full_plain: str,
    mode: str,
    model_label: str,
    max_state_chars: int,
    max_action_chars: int,
    max_memory_chars: int,
    fallback_reason: str,
) -> MemoryRecord:
    """Build a MemoryRecord that embeds (truncated) trajectory text for vector store ingest.

    Not used by ``maybe_summarize_and_write_experiences`` by default (full trajectories are not written to avoid
    store bloat). Kept for a future archival path (e.g. external files or a dedicated collection).
    """
    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    um0 = initial_utility_metadata()
    meta = {
        "dataset_item_id": dataset_item_id,
        "trajectory_index": trajectory_index,
        "memory_index_within_trajectory": 0,
        "extracted_by": model_label,
        "source": "experience_summarizer",
        "mode": mode,
        "experience_writeback_kind": "raw_trajectory_fallback",
        "fallback_reason": fallback_reason,
        "trajectory_chars": len(traj_full_plain),
        "trajectory_chars_for_prompt": 0,
        "trajectory_chars_full": len(traj_full_plain),
        **um0,
    }
    st = normalize_text("[raw_trajectory_fallback]", max_chars=max_state_chars)
    at = normalize_text(source_episode_id, max_chars=max_action_chars)
    mt = normalize_text(traj_full_plain, max_chars=max_memory_chars)
    return MemoryRecord(
        memory_id=str(uuid.uuid4()),
        task_name=task_name,
        item_id=int(trajectory_index),
        source_episode_id=source_episode_id,
        source_step=0,
        state_text=st,
        action_text=at,
        memory_text=mt,
        reward=float(episode_reward),
        success=bool(episode_success),
        created_step=None,
        created_at=now_ts,
        retrieval_count=0,
        last_used_step=None,
        metadata=meta,
        value=float(um0[UTILITY_SCORE_KEY]),
        value_source="utility_score_prior",
        value_update_step=None,
    )


def _memory_record_from_extracted_dict(
    *,
    memory: dict[str, Any],
    memory_idx: int,
    task_name: str,
    trajectory_index: int,
    dataset_item_id: Any,
    source_episode_id: str,
    episode_reward: float,
    episode_success: bool,
    mode: str,
    trajectory_chars_for_prompt: int,
    trajectory_chars_full: int,
    extractor_model_label: str,
    max_state_chars: int,
    max_action_chars: int,
    max_memory_chars: int,
    schema: str = "full",
    steps: Optional[List[Dict[str, Any]]] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> MemoryRecord | None:
    schema_n = (schema or "full").strip().lower()
    st_raw = memory["state_text"] if "state_text" in memory and memory["state_text"] is not None else ""
    at_raw = memory["action_text"] if "action_text" in memory and memory["action_text"] is not None else ""
    mt_raw = memory["memory_text"] if "memory_text" in memory and memory["memory_text"] is not None else ""
    memory_text = normalize_text(str(mt_raw), max_chars=max_memory_chars)
    if not memory_text:
        return None

    if "source_step" in memory and memory["source_step"] is not None:
        try:
            source_step = int(memory["source_step"])
        except (TypeError, ValueError):
            source_step = memory_idx + 1
    else:
        source_step = memory_idx + 1

    state_text = normalize_text(str(st_raw), max_chars=max_state_chars)
    action_text = normalize_text(str(at_raw), max_chars=max_action_chars)

    if schema_n == "compact" and steps is not None and tokenizer is not None:
        ordered = _ordered_active_steps(steps)
        if not state_text or not action_text:
            if 1 <= source_step <= len(ordered):
                _, step_one = ordered[source_step - 1]
                if not state_text:
                    state_text = normalize_text(
                        _state_text_from_step(step_one, tokenizer), max_chars=max_state_chars
                    )
                if not action_text:
                    action_text = normalize_text(
                        _action_text_from_step(step_one, tokenizer), max_chars=max_action_chars
                    )
        if not state_text:
            state_text = normalize_text(memory_text, max_chars=max_state_chars)
        if not action_text:
            action_text = normalize_text(memory_text, max_chars=max_action_chars)
    else:
        if not state_text or not action_text:
            return None

    if "metadata" in memory and isinstance(memory["metadata"], dict):
        metadata = dict(memory["metadata"])
    elif "metadata" in memory:
        metadata = {"raw_metadata": memory["metadata"]}
    else:
        metadata = {}
    metadata = {
        **metadata,
        "dataset_item_id": dataset_item_id,
        "trajectory_index": trajectory_index,
        "memory_index_within_trajectory": memory_idx,
        "extracted_by": extractor_model_label,
        "experience_writeback_kind": "llm_extracted_compact" if schema_n == "compact" else "llm_extracted",
        "extraction_schema": schema_n,
    }

    value = memory["value"] if "value" in memory else None
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None

    utility_meta = initial_utility_metadata()
    prior = float(utility_meta[UTILITY_SCORE_KEY])
    llm_value_meta: dict[str, Any] = {}
    if value is not None:
        llm_value_meta["llm_suggested_value"] = value

    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return MemoryRecord(
        memory_id=str(uuid.uuid4()),
        task_name=task_name,
        item_id=int(trajectory_index),
        source_episode_id=source_episode_id,
        source_step=source_step,
        state_text=state_text,
        action_text=action_text,
        memory_text=memory_text,
        reward=float(episode_reward),
        success=bool(episode_success),
        created_step=None,
        created_at=now_ts,
        retrieval_count=0,
        last_used_step=None,
        metadata={
            **metadata,
            **utility_meta,
            **llm_value_meta,
            "source": "experience_summarizer",
            "mode": mode,
            "trajectory_chars": trajectory_chars_for_prompt,
            "trajectory_chars_for_prompt": trajectory_chars_for_prompt,
            "trajectory_chars_full": trajectory_chars_full,
        },
        value=prior,
        value_source="utility_score_prior",
        value_update_step=None,
    )


def maybe_summarize_and_write_experiences(
    *,
    config: DictConfig,
    memory_manager: MemoryManager,
    tokenizer: PreTrainedTokenizer,
    actor_rollout_wg,
    total_batch_list: List[List[Dict[str, Any]]],
    total_infos: List[List[Dict[str, Any]]],
    episode_rewards: np.ndarray,
    episode_lengths: np.ndarray,
    success: Dict[str, np.ndarray],
    traj_uid: np.ndarray,
) -> int:
    """
    Implementation for ``MemoryManager.maybe_write_rollout_memories`` (also usable directly in tests).

    Requires ``env.memory.enabled``, ``env.memory.write_back``, and
    ``env.memory.experience_summarizer.mode`` in {``self``, ``teacher``}.
    The model response must be parseable as JSON with a ``memories`` array (see ``scripts/extract_memory_records.py``).
    If parsing yields no memories or every item fails validation, the same trajectory is inferred again until
    ``parse_max_attempts`` is reached (per trajectory).

    Episodes with no summarizer prompt after prepare are skipped (no vector insert). Full-trajectory archival is
    left to a future path; see ``_raw_trajectory_fallback_memory_record`` for a possible row shape.
    """
    if not experience_summarizer_active(config):
        return 0
    rm = memory_manager
    if not rm.enabled or rm.store is None:
        logger.debug("experience_summarizer: memory store unavailable; skip.")
        return 0

    mem = config.env.memory
    es = mem.experience_summarizer
    mode = str(es.mode).strip().lower()
    model_label = str(es.extractor_model_label)
    oa: DictConfig | None = None
    if mode == "teacher":
        oa = es.openai_api
        model_label = str(oa.model)
    only_pos = bool(mem.only_store_positive_reward)
    max_pairs = max(1, int(mem.max_pairs_per_episode))
    num_mem_cfg = int(es.num_memories_per_trajectory)
    num_memories_cap = max(1, min(num_mem_cfg, max_pairs))
    summarization_batch_size = int(es.summarization_batch_size)
    extraction_schema = _extraction_schema(es)
    system_prompt = _resolve_system_prompt(es, extraction_schema)
    trajectory_user_prompt_template = _resolve_trajectory_user_prompt_template(es, extraction_schema)
    apply_kw = OmegaConf.to_container(config.data["apply_chat_template_kwargs"], resolve=True) or {}

    max_state = int(mem.max_state_chars)
    max_act = int(mem.max_action_chars)
    max_mem = int(mem.max_memory_chars)

    batch_size = len(total_batch_list)
    trajectory_user_messages: List[str] = []
    pending_rollouts: List[Tuple[int, str, float, bool, str, int, Any]] = []
    pending_steps: List[List[Dict[str, Any]]] = []

    for i in range(batch_size):
        steps = total_batch_list[i]
        if not steps:
            continue
        r = float(np.asarray(episode_rewards[i], dtype=np.float64).reshape(-1)[0])
        if only_pos and r <= 0.0:
            continue
        succ = _episode_success(success, i)
        traj_s = str(traj_uid[i])
        dataset_item_id = _dataset_item_id_from_infos(total_infos, i)
        ep_len = float(episode_lengths[i]) if i < len(episode_lengths) else 0.0
        fmt_base = {
            "num_memories": num_memories_cap,
            "task_name": rm.task_name,
            "dataset_item_id": dataset_item_id,
            "trajectory_index": int(i),
            "episode_reward": r,
            "episode_length": ep_len,
            "traj_uid": traj_s,
            "success": int(succ),
        }
        traj_full_plain = _format_trajectory_plain(steps, tokenizer)
        traj_for_llm = _prepare_trajectory_text_for_summarizer(
            steps,
            tokenizer,
            config,
            es,
            trajectory_user_prompt_template=trajectory_user_prompt_template,
            template_format_kwargs=fmt_base,
            system_prompt=system_prompt,
            apply_kw=apply_kw if isinstance(apply_kw, dict) else None,
        )
        if not traj_for_llm or not traj_for_llm.strip():
            if traj_full_plain.strip():
                logger.info(
                    "experience_summarizer: env %s has no summarizer prompt after prepare; skip (no full-trajectory insert).",
                    i,
                )
            continue
        try:
            user_message = trajectory_user_prompt_template.format(
                trajectory_text=traj_for_llm,
                **fmt_base,
            )
        except KeyError as e:
            logger.warning("experience_summarizer: template placeholder missing: %s; skip env %s.", e, i)
            continue
        trajectory_user_messages.append(user_message)
        pending_rollouts.append((i, traj_s, r, succ, traj_for_llm, len(traj_full_plain), dataset_item_id))
        pending_steps.append(list(steps))

    if not trajectory_user_messages:
        return 0
    if mode not in ("self", "teacher"):
        raise ValueError(f"unsupported mode={mode}")

    parse_max_attempts = max(1, int(es.parse_max_attempts))

    n_pending = len(trajectory_user_messages)
    raw_outputs: List[str] = [""] * n_pending
    parse_attempts: List[int] = [0] * n_pending
    active: List[int] = list(range(n_pending))

    chat_prompts: List[str] | None = None
    if mode == "self":
        if actor_rollout_wg is None:
            logger.warning("experience_summarizer mode=self but actor_rollout_wg is None; skip.")
            return 0
        chat_prompts = [
            _build_chat_prompt(
                tokenizer, system_prompt, user_msg, apply_kw if isinstance(apply_kw, dict) else None
            )
            for user_msg in trajectory_user_messages
        ]

    try:
        while active:
            batch_out = _run_extraction_inference_for_indices(
                active,
                mode=mode,
                trajectory_user_messages=trajectory_user_messages,
                chat_prompts=chat_prompts,
                summarization_batch_size=summarization_batch_size,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                config=config,
                es=es,
                actor_rollout_wg=actor_rollout_wg,
                oa=oa,
            )
            for j, idx in enumerate(active):
                raw_outputs[idx] = batch_out[j]
                parse_attempts[idx] += 1

            next_active: List[int] = []
            for idx in active:
                recs = _memory_records_from_extractor_output(
                    raw_outputs[idx],
                    pending=pending_rollouts[idx],
                    task_name=rm.task_name,
                    mode=mode,
                    model_label=model_label,
                    max_pairs=max_pairs,
                    max_state_chars=max_state,
                    max_action_chars=max_act,
                    max_memory_chars=max_mem,
                    schema=extraction_schema,
                    steps=pending_steps[idx],
                    tokenizer=tokenizer,
                )
                if recs:
                    continue
                if parse_attempts[idx] >= parse_max_attempts:
                    logger.warning(
                        "experience_summarizer: trajectory index %s (env %s) still has no valid memories "
                        "after %s attempt(s); giving up.",
                        idx,
                        pending_rollouts[idx][0],
                        parse_attempts[idx],
                    )
                    continue
                next_active.append(idx)

            if next_active:
                logger.info(
                    "experience_summarizer: re-running extraction for %s trajectory(s) with empty/invalid parse "
                    "(attempt cap %s per trajectory).",
                    len(next_active),
                    parse_max_attempts,
                )
            active = next_active
    except Exception:
        logger.exception("experience_summarizer failed during model calls; no records written.")
        return 0

    records: List[MemoryRecord] = []
    for idx in range(n_pending):
        recs = _memory_records_from_extractor_output(
            raw_outputs[idx],
            pending=pending_rollouts[idx],
            task_name=rm.task_name,
            mode=mode,
            model_label=model_label,
            max_pairs=max_pairs,
            max_state_chars=max_state,
            max_action_chars=max_act,
            max_memory_chars=max_mem,
            schema=extraction_schema,
            steps=pending_steps[idx],
            tokenizer=tokenizer,
        )
        records.extend(recs)

    if not records:
        return 0
    try:
        rm.store.add_records(records)
        logger.info("experience_summarizer wrote %s memory record(s) (mode=%s).", len(records), mode)
    except Exception:
        logger.exception("experience_summarizer failed to add_records.")
        return 0
    return len(records)
