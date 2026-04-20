"""EvolveR-style memory utility counters (use/success + Laplace-smoothed score)."""

from __future__ import annotations

from typing import Any

import numpy as np

UTILITY_USE_KEY = "utility_use_count"
UTILITY_SUCC_KEY = "utility_succ_count"
UTILITY_LAPLACE_KEY = "utility_laplace"


def laplace_utility_score(c_use: int, c_succ: int) -> float:
    return (float(c_succ) + 1.0) / (float(c_use) + 2.0)


def collect_memory_ids_from_info_list(infos: list[dict[str, Any]] | None) -> set[str]:
    """Union of memory_id from every memory_event.retrieved along a trajectory."""
    ids: set[str] = set()
    if not infos:
        return ids
    for info in infos:
        if not isinstance(info, dict):
            continue
        ev = info.get("memory_event")
        if not ev or not isinstance(ev, dict):
            continue
        for item in ev.get("retrieved") or []:
            if not isinstance(item, dict):
                continue
            mid = item.get("memory_id")
            if mid:
                ids.add(str(mid))
    return ids


def episode_success_from_batch(success: dict[str, Any], index: int) -> bool:
    """Prefer ``success_rate`` when present (same convention as rollout success dict)."""
    if not success:
        return False
    sr = success.get("success_rate")
    if sr is not None:
        arr = np.asarray(sr)
        if arr.size > index:
            try:
                return bool(float(np.asarray(arr[index]).reshape(-1)[0]) != 0.0)
            except Exception:
                return bool(arr[index])
    for _k, arr in success.items():
        a = np.asarray(arr)
        if a.size <= index:
            continue
        try:
            return bool(np.asarray(a[index]).reshape(-1)[0] != 0)
        except Exception:
            return bool(a[index])
    return False
