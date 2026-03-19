from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    memory_id: str
    task_name: str
    item_id: int
    source_episode_id: str
    source_step: int
    state_text: str
    action_text: str
    memory_text: str
    reward: float
    success: bool
    created_step: int | str | None = None
    created_at: str | None = None
    retrieval_count: int = 0
    last_used_step: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    value: float | None = None
    value_source: str | None = None
    value_update_step: int | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        return cls(**payload)


@dataclass
class RetrievedMemory:
    memory_id: str
    score: float
    state_text: str
    action_text: str
    memory_text: str
    reward: float
    metadata: dict[str, Any] = field(default_factory=dict)
    value: float | None = None
    value_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEvent:
    event_type: str
    query_text: str
    state_text: str
    injected_text: str
    retrieved: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
