"""Match Hydra list-of-dicts phases against ``trainer_global_step`` (inclusive start, exclusive end)."""

from __future__ import annotations

from typing import Any, Optional

from omegaconf import OmegaConf


def match_phase_for_global_step(phases_cfg: Any, trainer_global_step: Optional[int]) -> Optional[dict]:
    """Return the first phase dict whose [global_step_start, global_step_end) contains ``trainer_global_step``.

    - ``global_step_end: null`` means unbounded above.
    - If ``phases_cfg`` is null/empty or ``trainer_global_step`` is None, returns None.
    """
    if trainer_global_step is None or phases_cfg is None:
        return None
    phases = OmegaConf.to_container(phases_cfg, resolve=True)
    if not isinstance(phases, list) or not phases:
        return None
    step = int(trainer_global_step)
    for ph in phases:
        if not isinstance(ph, dict):
            continue
        s0 = ph.get("global_step_start", 0)
        s0 = int(s0) if s0 is not None else 0
        s1 = ph.get("global_step_end")
        if step < s0:
            continue
        if s1 is not None and step >= int(s1):
            continue
        return ph
    return None
