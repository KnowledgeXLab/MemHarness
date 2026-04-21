"""Match Hydra list-of-dicts phases against ``trainer_global_step`` (inclusive start, exclusive end)."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Avoid spamming logs when match_phase is called every env step with a broken Hydra override.
_malformed_phases_repr_seen: set[str] = set()


def coerce_phases_to_dict_list(phases_cfg: Any) -> Optional[list[dict[str, Any]]]:
    """Turn Hydra/OmegaConf phase config into ``list[dict]``.

    Hydra CLI often misparses overrides like
    ``key='[{global_step_start:0,mode:fixed},...]'`` into a list of strings (e.g. ``['mode:fixed', ...]``),
    which makes phase matching a no-op. Prefer **JSON with double-quoted keys** in the shell, or YAML in a file.

    Returns ``None`` if ``phases_cfg`` is null/empty or unusable.
    """
    if phases_cfg is None:
        return None
    raw: Any = OmegaConf.to_container(phases_cfg, resolve=True)
    if isinstance(raw, str):
        raw_stripped = raw.strip()
        if not raw_stripped:
            return None
        try:
            raw = json.loads(raw_stripped)
        except json.JSONDecodeError:
            msg = raw_stripped[:200]
            if msg not in _malformed_phases_repr_seen:
                _malformed_phases_repr_seen.add(msg)
                logger.warning(
                    "Phases override looks like a string but is not valid JSON (%r). Ignoring phases.",
                    msg,
                )
            return None
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, Any]] = []
    non_dict = False
    for ph in raw:
        if isinstance(ph, dict):
            out.append(dict(ph))
        else:
            non_dict = True
    if non_dict or not out:
        rkey = repr(raw)
        if rkey not in _malformed_phases_repr_seen:
            _malformed_phases_repr_seen.add(rkey)
            logger.warning(
                "Invalid phases config: expected a list of objects, got %r. "
                "This usually means Hydra misparsed inline dicts from the CLI. "
                "Fix: pass JSON, e.g. "
                '\'[{\"global_step_start\":0,\"global_step_end\":30,\"mode\":\"fixed\"},'
                '{\"global_step_start\":30,\"global_step_end\":null,\"mode\":\"agentic\"}]\' '
                "or define phases in a YAML file. Phase-based overrides are disabled.",
                raw,
            )
        return None
    return out


def match_phase_for_global_step(phases_cfg: Any, trainer_global_step: Optional[int]) -> Optional[dict]:
    """Return the first phase dict whose [global_step_start, global_step_end) contains ``trainer_global_step``.

    - ``global_step_end: null`` means unbounded above.
    - If ``phases_cfg`` is null/empty or ``trainer_global_step`` is None, returns None.
    """
    if trainer_global_step is None or phases_cfg is None:
        return None
    phases = coerce_phases_to_dict_list(phases_cfg)
    if not phases:
        return None
    step = int(trainer_global_step)
    for ph in phases:
        s0 = ph.get("global_step_start", 0)
        s0 = int(s0) if s0 is not None else 0
        s1 = ph.get("global_step_end")
        if step < s0:
            continue
        if s1 is not None and step >= int(s1):
            continue
        return ph
    return None
