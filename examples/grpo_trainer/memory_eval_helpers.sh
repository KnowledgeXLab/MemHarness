# Shared helpers for eval / local-test launch scripts (Hydra overrides for memory prompts).
# Source from examples/grpo_trainer/*.sh — do not execute directly.

# Resolve ${REPO_ROOT}/data to a real directory (follows symlink, e.g. -> phwfile).
# Fails fast when compute nodes lack /mnt/phwfile or when `data` is a stray file.
resolve_repo_data_dir() {
  local link="${REPO_ROOT:?REPO_ROOT unset}/data"
  if [[ -L "${link}" ]]; then
    local resolved
    resolved="$(readlink -f "${link}" 2>/dev/null || true)"
    if [[ -z "${resolved}" || ! -d "${resolved}" ]]; then
      echo "[error] Broken/unavailable data symlink: ${link} -> $(readlink "${link}" 2>/dev/null)" >&2
      echo "[error] Ensure /mnt/phwfile is mounted on compute nodes (workspace/data -> phwfile)." >&2
      return 1
    fi
    printf '%s' "${resolved}"
  elif [[ -e "${link}" && ! -d "${link}" ]]; then
    echo "[error] ${link} exists but is not a directory (remove stray file?)." >&2
    return 1
  else
    mkdir -p "${link}"
    printf '%s' "$(readlink -f "${link}")"
  fi
}

# Per-bench placeholder parquet under ${REPO_DATA_DIR}/MemAdaptor/verl-agent/<bench>/text/.
# Aligns with exp_results under data/MemAdaptor/. Avoids concurrent --overwrite races across benches.
# Sets: DATA_ROOT, TRAIN_FILE, TEST_FILE, VAL_FILE (requires REPO_ROOT or REPO_DATA_DIR).
setup_verl_agent_text_data_paths() {
  local bench="${1:?bench name required (e.g. alfworld, webshop)}"
  if [[ -z "${REPO_DATA_DIR:-}" ]]; then
    REPO_DATA_DIR="$(resolve_repo_data_dir)" || return 1
  fi
  DATA_ROOT="${REPO_DATA_DIR}/MemAdaptor/verl-agent/${bench}"
  TRAIN_FILE="${DATA_ROOT}/text/train.parquet"
  TEST_FILE="${DATA_ROOT}/text/test.parquet"
  VAL_FILE="${TEST_FILE}"
  echo "[log] verl-agent data bench=${bench} DATA_ROOT=${DATA_ROOT}"
}

# Hydra: quote a string value (commas / spaces safe).
hydra_quote_string() {
  local s="${1:-}"
  s="${s//\'/\'\\\'\'}"
  printf "'%s'" "$s"
}

# Resolve RETRIEVAL_INSTRUCTION_PROMPT from env vars (mutates caller scope if sourced).
#   USE_GENERAL_MODEL_RETRIEVAL_HINT=1  → built-in general-model wording (agentic + memory only)
#   RETRIEVAL_INSTRUCTION_PROMPT        → explicit override
#   RETRIEVAL_INSTRUCTION_PROMPT_FILE   → load override from file
resolve_retrieval_instruction_prompt() {
  if [ -z "${RETRIEVAL_INSTRUCTION_PROMPT:-}" ] && [ -n "${RETRIEVAL_INSTRUCTION_PROMPT_FILE:-}" ]; then
    if [ ! -f "${RETRIEVAL_INSTRUCTION_PROMPT_FILE}" ]; then
      echo "[error] RETRIEVAL_INSTRUCTION_PROMPT_FILE not found: ${RETRIEVAL_INSTRUCTION_PROMPT_FILE}" >&2
      exit 1
    fi
    RETRIEVAL_INSTRUCTION_PROMPT="$(cat "${RETRIEVAL_INSTRUCTION_PROMPT_FILE}")"
  fi

  if [ -z "${RETRIEVAL_INSTRUCTION_PROMPT:-}" ] \
    && [ "${USE_GENERAL_MODEL_RETRIEVAL_HINT:-0}" = "1" ] \
    && [ "${MEMORY_ENABLED:-False}" = "True" ] \
    && [ "${RETRIEVAL_MODE:-agentic}" = "agentic" ]; then
    RETRIEVAL_INSTRUCTION_PROMPT='You have access to a memory bank of past experiences from similar tasks. When you are uncertain how to proceed, prefer retrieving relevant memory for guidance before taking an environment action. To retrieve memory, output exactly one search query wrapped by {open_tag} and {close_tag}. Do not output both a memory retrieval query and an action in the same response. After memory is returned, you will act in the next response.'
  fi
}

# Append env.memory.retrieval_instruction_prompt to MEMORY_CLI when set (agentic eval).
append_retrieval_instruction_cli() {
  resolve_retrieval_instruction_prompt
  if [ "${MEMORY_ENABLED:-False}" != "True" ]; then
    return 0
  fi
  if [ "${RETRIEVAL_MODE:-agentic}" != "agentic" ]; then
    return 0
  fi
  if [ -z "${RETRIEVAL_INSTRUCTION_PROMPT:-}" ]; then
    return 0
  fi
  MEMORY_CLI+=("env.memory.retrieval_instruction_prompt=$(hydra_quote_string "${RETRIEVAL_INSTRUCTION_PROMPT}")")
  echo "[log] env.memory.retrieval_instruction_prompt=${RETRIEVAL_INSTRUCTION_PROMPT}"
}
