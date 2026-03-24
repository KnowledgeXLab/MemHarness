from collections import Counter
from typing import Any
import json




def summarize_counts(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels: list[str] = []
    for r in results:
        if r.get("error"):
            labels.append("__error__")
            continue
        raw = r.get("judge_raw")
        if isinstance(raw, dict) and raw.get("primary_label"):
            labels.append(str(raw["primary_label"]))
        else:
            labels.append("__error__")
    cnt = Counter(labels)
    return {"primary_label_counts": dict(cnt), "total": len(results)}


data_path = "data/exp_results/MemAdaptor/pre_exp/alfworld/Qwen2.5-1.5B-Instruct-with_agentic_memory-retrieve_memory_text/train_traj/0.mem_judge-only-mem_fail_trajs-old.jsonl"

data = [json.loads(line) for line in open(data_path)]
summary = summarize_counts(data)
print(summary)