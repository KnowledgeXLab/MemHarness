#!/usr/bin/env bash
#SBATCH --job-name=e05-webshop-probe
#SBATCH --partition=DataFrontier_Explore
#SBATCH --quotatype=reserved
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=logs/mem_adaptor/webshop/probe/counterfactual_probe_%j.out
#SBATCH --error=logs/mem_adaptor/webshop/probe/counterfactual_probe_%j.err

# Offline counterfactual adaptor probe (GPU compute node).
#
# Submit from repo root:
#   mkdir -p logs/mem_adaptor/webshop/probe
#   sbatch scripts/run_counterfactual_probe.sh


set -euo pipefail

domain="webshop"  # webshop or alfworld 

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p logs/mem_adaptor/${domain}/probe



# --- Edit defaults (or pass via sbatch --export) ---
# MODEL_PATH="models/save_models/mem_adaptor/alfworld/train_adaptor-same-7B-cold_start_20260706_epoch2-with_agentic_memory-retrieve_memory_text-self_distill/best_val/global_step_170/actor/huggingface"
MODEL_PATH="models/save_models/mem_adaptor/webshop/train_adaptor-same-7B-cold_start_20260706_epoch2-with_agentic_memory-retrieve_memory_text-self_distill/global_step_80/actor/huggingface"
PAIRS_JSONL="data/MemAdaptor/exp_results/${domain}/probe/counterfactual_pairs.jsonl"
OUTPUT_JSONL="data/MemAdaptor/exp_results/${domain}/probe/counterfactual_probe_results.jsonl"
METRICS_JSON="data/MemAdaptor/exp_results/${domain}/probe/counterfactual_probe_metrics.json"
BATCH_SIZE="${BATCH_SIZE:-32}"
STRICT_EMPTY_REJECT="${STRICT_EMPTY_REJECT:-true}"

echo "[counterfactual-probe] REPO_ROOT=${REPO_ROOT}"
echo "[counterfactual-probe] MODEL_PATH=${MODEL_PATH}"
echo "[counterfactual-probe] PAIRS_JSONL=${PAIRS_JSONL}"
echo "[counterfactual-probe] STRICT_EMPTY_REJECT=${STRICT_EMPTY_REJECT}"
echo "[counterfactual-probe] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

STRICT_EMPTY_ARGS=()
if [[ "${STRICT_EMPTY_REJECT,,}" == "true" || "${STRICT_EMPTY_REJECT}" == "1" ]]; then
  STRICT_EMPTY_ARGS=(--strict_empty_reject)
else
  STRICT_EMPTY_ARGS=(--no-strict_empty_reject)
fi

exec python scripts/run_counterfactual_probe.py \
  --model_path "${MODEL_PATH}" \
  --pairs_jsonl "${PAIRS_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --metrics_json "${METRICS_JSON}" \
  --batch_size "${BATCH_SIZE}" \
  "${STRICT_EMPTY_ARGS[@]}"
