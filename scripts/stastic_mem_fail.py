"""汇总 ``mem_fail_judge.py`` 输出的 JSONL（支持多标签 ``labels``）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 与 ``python scripts/stastic_mem_fail.py`` 同目录的 mem_fail_judge
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem_fail_judge import summarize_counts  # noqa: E402


def main() -> None:
    data_path = "data/exp_results/MemAdaptor/pre_exp/alfworld/gpt-4.1-nano-with_agentic_memory-retrieve_memory_text/train_traj/0.mem_judge-only-mem_fail_trajs-1.jsonl"
    data: list[dict[str, Any]] = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
    summary = summarize_counts(data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
