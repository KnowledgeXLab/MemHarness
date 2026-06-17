#!/usr/bin/env bash
# Run memory_db_export_rebuild.py inside Apptainer (CentOS7 login nodes lack GLIBC for Milvus Lite).
#
# Usage (same args as the python script):
  # bash scripts/run_memory_db_export_rebuild.sh --action export_only \
  #   --source_store_dir data/MemAdaptor/exp_results/alfworld/train_adaptor-same-7B-cold_start_20260519_epoch1-1-with_agentic_memory-retrieve_memory_text-self_distill/memory_vdb \
  #   --source_retrieve_key memory_text \
  #   --jsonl_path data/MemAdaptor/exp_results/alfworld/train_adaptor-same-7B-cold_start_20260519_epoch1-1-with_agentic_memory-retrieve_memory_text-self_distill/memory_vdb/mem_export.jsonl
#
# Optional env overrides:
#   APPTAINER_SIF=/path/to/glibc_ubuntu22.sif
#   CONDA_SH=~/miniconda3/etc/profile.d/conda.sh
#   CONDA_ENV=verl-agent
#   MEMADAPTOR_ROOT=/path/to/MemAdaptor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MEMADAPTOR_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

APPTAINER_SIF="${APPTAINER_SIF:-/mnt/petrelfs/wurong/glibc_ubuntu22.sif}"
CONDA_SH="${CONDA_SH:-/mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-verl-agent}"
APPTAINER_BINDS="${APPTAINER_BINDS:-/mnt:/mnt}"

if [[ ! -f "${APPTAINER_SIF}" ]]; then
  echo "ERROR: Apptainer image not found: ${APPTAINER_SIF}" >&2
  echo "Set APPTAINER_SIF to glibc_ubuntu22.sif (same as train_alfworld-adaptor-local-test.sh)." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/scripts/memory_db_export_rebuild.py" ]]; then
  echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like MemAdaptor." >&2
  exit 1
fi

BIND_OPTS=""
for part in ${APPTAINER_BINDS}; do
  part="${part// /}"
  [[ -z "${part}" ]] && continue
  BIND_OPTS+=" -B ${part}"
done

exec apptainer exec --nv${BIND_OPTS} "${APPTAINER_SIF}" bash -lc "
set -eo pipefail
set +u
source \"${CONDA_SH}\"
conda activate \"${CONDA_ENV}\"
set -u
cd \"${REPO_ROOT}\"
export PYTHONPATH=\"${REPO_ROOT}:\${PYTHONPATH:-}\"
exec python3 scripts/memory_db_export_rebuild.py \"\$@\"
" _ "$@"
