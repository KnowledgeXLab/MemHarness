#!/usr/bin/env python3
"""Run offline counterfactual adaptor probe on probe pairs JSONL.

Reuses ``mem_adaptor_rollout`` prompt construction and output normalization.
Edit MODEL_PATH / PAIRS_JSONL defaults below, or pass CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Sequence

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_system.memory.mem_adaptor_rollout import (
    _build_adaptor_prompt,
    _normalize_adaptor_output,
    adaptor_apply_chat_template_kwargs,
)
from agent_system.memory.mem_adaptor_training import _normalize_principle_text

# --- Edit these defaults ---
DEFAULT_MODEL_PATH = "models/save_models/mem_adaptor/alfworld/train_adaptor-same-7B-cold_start_20260706_epoch2-with_agentic_memory-retrieve_memory_text-self_distill/best_val/global_step_170/actor/huggingface"
DEFAULT_PAIRS_JSONL = "data/MemAdaptor/exp_results/alfworld/probe/counterfactual_pairs.jsonl"
DEFAULT_CONFIG_YAML = "verl/trainer/config/ppo_trainer.yaml"
DEFAULT_OUTPUT_JSONL = "data/MemAdaptor/exp_results/alfworld/probe/counterfactual_probe_results.jsonl"
DEFAULT_METRICS_JSON = "data/MemAdaptor/exp_results/alfworld/probe/counterfactual_probe_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="HF adaptor checkpoint directory.")
    parser.add_argument("--pairs_jsonl", default=DEFAULT_PAIRS_JSONL, help="Input from build_counterfactual_probe_pairs.py")
    parser.add_argument("--config_yaml", default=DEFAULT_CONFIG_YAML, help="PPO yaml for mem_adaptor prompt template.")
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--metrics_json", default=DEFAULT_METRICS_JSON)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Greedy decode by default (recommended for probe).",
    )
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of pairs (0 = all).")
    parser.add_argument(
        "--strict_empty_reject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, reject metrics count only explicit <EMPTY> (or configured empty markers) "
            "in the model completion. Prompt-echo / malformed generations are not treated as reject."
        ),
    )
    return parser.parse_args()


def _load_pairs(path: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("task", "s_old", "p_old", "s_plus", "s_minus", "memory_id"):
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"Invalid pair row (missing {key}): {row.get('memory_id', '')}")
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No pairs loaded from {path}")
    return rows


def _classify_accept(norm: str, p_old: str) -> str:
    if _normalize_principle_text(norm) == _normalize_principle_text(p_old):
        return "unchanged"
    return "adapted"


def _extract_adaptor_completion(raw: str) -> str:
    """Best-effort completion text after chat-template ``assistant`` turn."""
    text = (raw or "").strip()
    if not text:
        return ""
    parts = re.split(r"\nassistant\b", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[1].strip()
    return text


def _is_true_empty_reject(raw: str, empty_markers: Sequence[str]) -> bool:
    """Reject only when completion is exactly an configured empty marker (e.g. ``<EMPTY>``)."""
    completion = _extract_adaptor_completion(raw)
    if not completion:
        return False
    low = completion.lower()
    for marker in empty_markers:
        if completion == marker or low == marker.lower():
            return True
    return False


def _row_reject_flag(row: dict[str, Any], *, strict_empty_reject: bool) -> bool:
    if strict_empty_reject:
        return bool(row.get("is_true_empty_reject"))
    return bool(row.get("is_reject"))


def _row_label(row: dict[str, Any], *, strict_empty_reject: bool, p_old: str) -> str:
    if _row_reject_flag(row, strict_empty_reject=strict_empty_reject):
        return "reject"
    norm = str(row.get("norm_output") or "")
    return _classify_accept(norm, p_old)


def _build_prompts(
    pairs: list[dict[str, Any]],
    *,
    config,
    tokenizer,
    chat_kw: dict,
) -> tuple[list[str], list[dict[str, Any]]]:
    ma = config.mem_adaptor
    meta_rows: list[dict[str, Any]] = []
    prompts: list[str] = []
    for pair in pairs:
        for variant, s_curr in (("plus", pair["s_plus"]), ("minus", pair["s_minus"])):
            prompt = _build_adaptor_prompt(
                tokenizer,
                ma,
                task=str(pair["task"]),
                s_curr=str(s_curr),
                s_old=str(pair["s_old"]),
                p_old=str(pair["p_old"]),
                apply_chat_template_kwargs=chat_kw,
            )
            prompts.append(prompt)
            meta_rows.append(
                {
                    "memory_id": pair["memory_id"],
                    "variant": variant,
                    "task": pair["task"],
                    "s_old": pair["s_old"],
                    "p_old": pair["p_old"],
                    "s_curr": s_curr,
                    "flip": pair.get("flip", ""),
                }
            )
    return prompts, meta_rows


@torch.inference_mode()
def _generate_batches(
    model,
    tokenizer,
    prompts: list[str],
    *,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs: list[str] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="adaptor infer"):
        chunk = prompts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p
        generated = model.generate(**enc, **gen_kwargs)
        input_lens = enc["attention_mask"].sum(dim=1).tolist()
        for i, seq in enumerate(generated):
            new_tokens = seq[int(input_lens[i]) :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text)
    return outputs


def _compute_metrics(results: list[dict[str, Any]], *, strict_empty_reject: bool) -> dict[str, Any]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        mid = str(row["memory_id"])
        by_pair.setdefault(mid, {})[str(row["variant"])] = row

    def _summarize(*, strict: bool) -> dict[str, Any]:
        plus_rows: list[dict[str, Any]] = []
        minus_rows: list[dict[str, Any]] = []
        paired_correct = 0
        paired_both_reject = 0
        paired_both_accept = 0
        complete_pairs = 0

        for variants in by_pair.values():
            if "plus" not in variants or "minus" not in variants:
                continue
            complete_pairs += 1
            plus = variants["plus"]
            minus = variants["minus"]
            plus_rows.append(plus)
            minus_rows.append(minus)
            plus_reject = _row_reject_flag(plus, strict_empty_reject=strict)
            minus_reject = _row_reject_flag(minus, strict_empty_reject=strict)
            if minus_reject and not plus_reject:
                paired_correct += 1
            if plus_reject and minus_reject:
                paired_both_reject += 1
            if (not plus_reject) and (not minus_reject):
                paired_both_accept += 1

        def _reject_rate(rows: list[dict[str, Any]]) -> float:
            if not rows:
                return 0.0
            return float(sum(1 for r in rows if _row_reject_flag(r, strict_empty_reject=strict)) / len(rows))

        reject_plus = _reject_rate(plus_rows)
        reject_minus = _reject_rate(minus_rows)
        accept_plus = 1.0 - reject_plus
        accept_minus = 1.0 - reject_minus

        def _accept_breakdown(rows: list[dict[str, Any]]) -> dict[str, float]:
            if not rows:
                return {"unchanged": 0.0, "adapted": 0.0, "reject": 0.0}
            n = len(rows)
            counts = {"unchanged": 0, "adapted": 0, "reject": 0}
            for r in rows:
                label = _row_label(r, strict_empty_reject=strict, p_old=str(r["p_old"]))
                counts[label] += 1
            return {k: float(v / n) for k, v in counts.items()}

        return {
            "n_pairs_complete": complete_pairs,
            "reject_rate_plus": reject_plus,
            "reject_rate_minus": reject_minus,
            "accept_rate_plus": accept_plus,
            "accept_rate_minus": accept_minus,
            "delta_reject": reject_minus - reject_plus,
            "delta_accept": accept_plus - accept_minus,
            "paired_correct_rate": float(paired_correct / complete_pairs) if complete_pairs else 0.0,
            "paired_both_reject_rate": float(paired_both_reject / complete_pairs) if complete_pairs else 0.0,
            "paired_both_accept_rate": float(paired_both_accept / complete_pairs) if complete_pairs else 0.0,
            "plus_label_rates": _accept_breakdown(plus_rows),
            "minus_label_rates": _accept_breakdown(minus_rows),
        }

    primary = _summarize(strict=strict_empty_reject)
    alternate = _summarize(strict=not strict_empty_reject)
    true_empty = primary if strict_empty_reject else alternate
    reject_metric_mode = "true_empty" if strict_empty_reject else "standard"
    alternate_prefix = "standard_" if strict_empty_reject else "true_empty_"
    metrics = {
        **primary,
        "reject_metric_mode": reject_metric_mode,
        "strict_empty_reject": bool(strict_empty_reject),
        "true_empty_plus_label_rates": true_empty["plus_label_rates"],
        "true_empty_minus_label_rates": true_empty["minus_label_rates"],
        f"{alternate_prefix}reject_rate_plus": alternate["reject_rate_plus"],
        f"{alternate_prefix}reject_rate_minus": alternate["reject_rate_minus"],
        f"{alternate_prefix}delta_reject": alternate["delta_reject"],
        f"{alternate_prefix}paired_correct_rate": alternate["paired_correct_rate"],
    }
    return metrics


def main() -> None:
    args = parse_args()
    model_path = os.path.abspath(args.model_path)
    pairs_path = os.path.abspath(args.pairs_jsonl)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not os.path.isfile(pairs_path):
        raise FileNotFoundError(f"pairs_jsonl not found: {pairs_path}")

    config_path = os.path.join(PROJECT_ROOT, args.config_yaml)
    config = OmegaConf.load(config_path)
    ma = config.mem_adaptor
    empty_markers = list(ma["empty_output_markers"])

    pairs = _load_pairs(pairs_path, args.limit)
    print(f"Loaded {len(pairs)} pairs -> {len(pairs) * 2} adaptor calls", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.device == "auto":
        device_map = "auto" if torch.cuda.is_available() else None
    else:
        device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if args.device != "auto" and device_map is None:
        model = model.to(args.device)
    model.eval()

    chat_kw = adaptor_apply_chat_template_kwargs(config, ma)
    prompts, meta_rows = _build_prompts(pairs, config=config, tokenizer=tokenizer, chat_kw=chat_kw)
    raw_outputs = _generate_batches(
        model,
        tokenizer,
        prompts,
        batch_size=max(1, args.batch_size),
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    results: list[dict[str, Any]] = []
    for meta, raw in zip(meta_rows, raw_outputs):
        raw_text = str(raw)
        norm, is_reject = _normalize_adaptor_output(raw_text, empty_markers)
        is_true_empty = _is_true_empty_reject(raw_text, empty_markers)
        label = _row_label(
            {"is_reject": is_reject, "is_true_empty_reject": is_true_empty, "norm_output": norm},
            strict_empty_reject=args.strict_empty_reject,
            p_old=str(meta["p_old"]),
        )
        results.append(
            {
                **meta,
                "raw_output": raw,
                "norm_output": norm,
                "adaptor_completion": _extract_adaptor_completion(raw_text),
                "is_reject": bool(is_reject),
                "is_true_empty_reject": bool(is_true_empty),
                "label": label,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = _compute_metrics(results, strict_empty_reject=args.strict_empty_reject)
    metrics.update(
        {
            "model_path": model_path,
            "pairs_jsonl": pairs_path,
            "n_calls": len(results),
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
        }
    )
    with open(args.metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\n=== Counterfactual Probe Metrics ===", flush=True)
    print(f"reject_metric_mode: {metrics['reject_metric_mode']}", flush=True)
    for key in (
        "n_pairs_complete",
        "reject_rate_plus",
        "reject_rate_minus",
        "delta_reject",
        "paired_correct_rate",
        "plus_label_rates",
        "minus_label_rates",
    ):
        print(f"{key}: {metrics[key]}", flush=True)
    alt_prefix = "standard_" if args.strict_empty_reject else "true_empty_"
    print(
        f"{alt_prefix}reject_rate_plus: {metrics[f'{alt_prefix}reject_rate_plus']}",
        flush=True,
    )
    print(
        f"{alt_prefix}reject_rate_minus: {metrics[f'{alt_prefix}reject_rate_minus']}",
        flush=True,
    )
    print(f"{alt_prefix}delta_reject: {metrics[f'{alt_prefix}delta_reject']}", flush=True)
    print(
        f"{alt_prefix}paired_correct_rate: {metrics[f'{alt_prefix}paired_correct_rate']}",
        flush=True,
    )
    print(f"true_empty_plus_label_rates: {metrics['true_empty_plus_label_rates']}", flush=True)
    print(f"true_empty_minus_label_rates: {metrics['true_empty_minus_label_rates']}", flush=True)
    print(f"\nWrote results -> {os.path.abspath(args.output_jsonl)}", flush=True)
    print(f"Wrote metrics -> {os.path.abspath(args.metrics_json)}", flush=True)


if __name__ == "__main__":
    main()
