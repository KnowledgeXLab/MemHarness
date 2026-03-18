from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
import concurrent.futures
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """You are an expert agent-memory extraction system.

Your job is to read one successful trajectory from an interactive agent benchmark and extract a small set of reusable memories.

The extracted memories must satisfy all requirements:
1. Each memory should correspond to one local decision point or one short reusable principle grounded in the trajectory.
2. A memory must be useful for retrieval in a future similar state.
3. Do not summarize the whole trajectory into one global high-level paragraph.
4. Prefer step-level or subgoal-level memories.
5. Do not invent facts that are not supported by the trajectory.
6. The output must be valid JSON only, with no markdown fences and no extra explanation.
"""


USER_PROMPT_TEMPLATE = """Extract at most {num_memories} reusable memories from the following successful trajectory.

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

Trajectory metadata:
- dataset_item_id: {dataset_item_id}
- trajectory_index: {trajectory_index}

Trajectory:
{trajectory_text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract memory records from successful AgentGym trajectories with an OpenAI-compatible LLM.")
    parser.add_argument("--input_dir", default="data/AgentGym/AgentTraj-L", help="Directory containing the source trajectory json files.")
    parser.add_argument("--benchmarks", default=["alfworld", "webshop"], help="[alfworld, webshop, sciworld]")
    parser.add_argument("--base_url", default="http://35.220.164.252:3888/v1/", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="gpt-5.1", help="LLM model name for extraction.")
    parser.add_argument("--start_index", type=int, default=0, help="Start trajectory index.")
    parser.add_argument("--max_trajectories", type=int, default=-1, help="How many trajectories to process per benchmark.")
    parser.add_argument("--num_memories_per_trajectory", type=int, default=2, help="Upper bound of memories extracted from each trajectory.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of concurrent LLM calls.")
    
    return parser.parse_args()


def load_trajectories(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of trajectories in {path}")
    return payload


def trajectory_to_text(trajectory: dict[str, Any]) -> str:
    conversations = trajectory.get("conversations", [])
    lines: list[str] = []
    action_step = 0

    for idx, turn in enumerate(conversations):
        role = turn.get("from", "unknown")
        content = str(turn["value"]).strip()
        if role == "gpt" and turn.get("loss") is True:
            action_step += 1
            prefix = f"[Step {action_step}] agent"
        elif role == "human":
            prefix = f"[Turn {idx}] environment"
        else:
            prefix = f"[Turn {idx}] {role}"
        lines.append(f"{prefix}: {content}")

    text = "\n\n".join(lines)
    return text


def extract_json_object(text: str) -> dict[str, Any]:
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


def build_client(base_url: str, api_key: str, timeout: int) -> OpenAI:
    http_client = httpx.Client(verify=False, trust_env=False, timeout=float(timeout))
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def call_extractor(
    client: OpenAI,
    model: str,
    task_name: str,
    dataset_item_id: Any,
    trajectory_index: int,
    trajectory_text: str,
    num_memories: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        num_memories=num_memories,
        task_name=task_name,
        dataset_item_id=dataset_item_id,
        trajectory_index=trajectory_index,
        trajectory_text=trajectory_text,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    return extract_json_object(content)


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def build_record(
    memory: dict[str, Any],
    task_name: str,
    trajectory_index: int,
    dataset_item_id: Any,
    memory_idx: int,
    model: str,
) -> dict[str, Any] | None:
    state_text = normalize_text(memory.get("state_text"))
    action_text = normalize_text(memory.get("action_text"))
    memory_text = normalize_text(memory.get("memory_text"))
    if not state_text or not action_text or not memory_text:
        return None

    metadata = memory.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    metadata = {
        **metadata,
        "dataset_item_id": dataset_item_id,
        "trajectory_index": trajectory_index,
        "memory_index_within_trajectory": memory_idx,
        "extracted_by": model,
    }

    source_step = memory.get("source_step", memory_idx + 1)
    try:
        source_step = int(source_step)
    except (TypeError, ValueError):
        source_step = memory_idx + 1

    value = memory.get("value", None)
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
    # 转化成人类可读的时间戳
    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    source_episode_id = str(dataset_item_id) if dataset_item_id is not None else f"{task_name}_{trajectory_index}"

    return {
        "memory_id": str(uuid.uuid4()),
        "task_name": task_name,
        "item_id": trajectory_index,
        "source_episode_id": source_episode_id,
        "source_step": source_step,
        "state_text": state_text,
        "action_text": action_text,
        "memory_text": memory_text,
        "reward": 1.0,
        "success": True,
        "created_step": None,
        "created_at": now_ts,
        "retrieval_count": 0,
        "last_used_step": None,
        "metadata": metadata,
        "value": 0.0, # default value is 0.0
        "value_source": None,
        "value_update_step": None,
    }


def ensure_output_path(path: str, overwrite: bool) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)


def process_benchmark(bench: str, args: argparse.Namespace) -> None:
    input_path = os.path.join(args.input_dir, f"{bench}_train.json")
    output_path = os.path.join(args.input_dir, f"{bench}_train_memory_records-{args.model}.jsonl")

    if not os.path.exists(input_path):
        print(f"Input file not found for {bench}: {input_path}")
        return

    try:
        ensure_output_path(output_path, overwrite=args.overwrite)
    except FileExistsError as e:
        print(e)
        return

    print(f"Processing benchmark: {bench} from {input_path}")
    trajectories = load_trajectories(input_path)
    end_index = min(len(trajectories), args.start_index + args.max_trajectories) if args.max_trajectories > 0 else len(trajectories)
    selected = trajectories[args.start_index:end_index]

    def process_trajectory(offset: int, trajectory: dict[str, Any]) -> list[dict[str, Any]]:
        client = build_client(base_url=args.base_url, api_key= "sk-5QyBNRgeFFiX6sY1aooYjvtygjNelFW87I6ziXkE6mP6tVeH", timeout=args.timeout)
        trajectory_index = args.start_index + offset
        dataset_item_id = trajectory.get("item_id", trajectory_index)
        trajectory_text = trajectory_to_text(trajectory)
        
        if not trajectory_text:
            return []

        try:
            extracted = call_extractor(
                client=client,
                model=args.model,
                task_name=bench,
                dataset_item_id=dataset_item_id,
                trajectory_index=trajectory_index,
                trajectory_text=trajectory_text,
                num_memories=args.num_memories_per_trajectory,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            memories = extracted["memories"]
            if not isinstance(memories, list):
                return []

            batch_records = []
            for memory_idx, memory in enumerate(memories):
                if not isinstance(memory, dict):
                    continue
                record = build_record(
                    memory=memory,
                    task_name=bench,
                    trajectory_index=trajectory_index,
                    dataset_item_id=dataset_item_id,
                    memory_idx=memory_idx,
                    model=args.model,
                )
                if record is not None:
                    batch_records.append(record)
            
            return batch_records
        except Exception as e:
            print(f"[{bench}] Error processing trajectory {trajectory_index}: {e}")
            return []

    records: list[dict[str, Any]] = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_trajectory, offset, traj): offset
            for offset, traj in enumerate(selected)
        }
        
        with tqdm(total=len(selected), desc=f"Extracting {bench}", unit="traj") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                records.extend(result)
                pbar.update(1)
                pbar.set_postfix({"memories": len(records)})

    # Sort records sequentially by trajectory_index and source_step
    records.sort(key=lambda x: (x["item_id"], x["source_step"]))

    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "task_name": bench,
                "input_path": input_path,
                "output_path": output_path,
                "num_trajectories_processed": len(selected),
                "num_records_written": len(records),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    benchmarks = args.benchmarks
    
    if not benchmarks:
        print("No benchmarks specified.")
        return

    for bench in benchmarks:
        process_benchmark(bench, args)


if __name__ == "__main__":
    main()
