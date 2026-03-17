from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    memory_id: str  # Unique id of this memory entry.
    task_name: str  # Task/benchmark name, e.g. sciworld or babyai.
    item_id: int  # Concrete environment instance id used in rollout.
    source_episode_id: str  # Episode identifier that produced this memory.
    source_step: int  # Step index inside the source episode.
    state_text: str  # State/observation text paired with the action.
    action_text: str  # Assistant action taken under the source state.
    memory_text: str  # Textual memory payload injected during retrieval.
    reward: float  # Final episode reward associated with the memory.
    success: bool  # Whether the source episode is considered successful.
    created_step: int | str | None = None  # Trainer global step or eval batch tag.
    created_at: int | None = None  # The timestamp when the memory was created.
    retrieval_count: int = 0  # Number of times this memory was retrieved in-process.
    last_used_step: int | str | None = None  # Reserved field for future usage tracking.
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra debug or provenance info.
    value: float | None = None   # The value of the memory record. eg: success-rate proxy, learned score
    value_source: str | None = None  # The source of the value. eg: heuristic, judge, critic
    value_update_step: int | str | None = None  # The step when the value was updated. To clean up old memories.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        return cls(**payload)


@dataclass
class RetrievedMemory:
    memory_id: str  # Source memory id.
    score: float  # Retrieval similarity score for the current query.
    state_text: str  # Historical state matched by retrieval.
    action_text: str  # Historical action associated with that state.
    memory_text: str  # Final memory text available for prompt injection.
    reward: float  # Reward of the episode this memory came from.
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra source info for analysis.
    value: float | None = None  # Optional memory value for future reranking or cleanup.
    value_source: str | None = None  # Where the value came from, e.g. heuristic or critic.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEvent:
    event_type: str  # Type of memory event, currently retrieval.
    round_idx: int  # Interaction round when the event happened.
    query_text: str  # Query text used to retrieve memories.
    state_text: str  # Current state text that triggered retrieval.
    injected_text: str  # Prompt text injected into the conversation.
    retrieved: list[dict[str, Any]] = field(default_factory=list)  # Serialized retrieved memories.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
