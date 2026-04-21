#!/usr/bin/env bash
# 周期性将本地 W&B 离线 run 目录同步到云端。
#
# 用法:
#   export WANDB_API_KEY=...   # 与训练时一致
#   bash scripts/wandb_offline_sync_loop.sh          # 间隔默认 3 分钟
#   bash scripts/wandb_offline_sync_loop.sh 10       # 每 10 分钟一轮
#
# 同步哪些目录: 只改本脚本上方「配置区」即可，无需在命令行传文件路径。
#   优先级: 内联数组 WANDB_OFFLINE_SYNC_RUN_DIRS（非空）> 列表文件
#           WANDB_OFFLINE_SYNC_LIST_FILE > 扫描 WANDB_OFFLINE_SYNC_SCAN_ROOT
#
# 列表文件规则（与内联无关、仅当用文件时）:
#   - 空行、仅空白行忽略；以 # 开头的整行视为注释
#   - 其余行去掉首尾空白后为目录路径；不存在或非目录则跳过并打日志
#
# 说明:
#   - 训练侧保持 WANDB_MODE=offline 即可；本脚本对每个目录调用 wandb sync，并临时设 WANDB_MODE=online。
#   - 需当前环境已安装 wandb 且能访问 W&B 服务。
#   - 同一 run 多次 sync 一般可接受；若出现重复 run，可改由 wandb 网页端处理或仅同步尚未上传的目录。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 配置区（按需修改；改完后 kill 旧进程再重新启动本脚本即可）
# ---------------------------------------------------------------------------

# 方式 1 — 在数组里直接写 offline-run-* 的绝对路径（推荐少量 run；非空则不再读列表文件）
WANDB_OFFLINE_SYNC_RUN_DIRS=(
  # 示例:
  # "/abs/path/to/repo/wandb/offline-run-20260421_112141-xxxxxxxx"
  "/mnt/petrelfs/wurong/workspace/MemAdaptor/wandb/offline-run-20260421_095915-uiw8ber1"
  "/mnt/petrelfs/wurong/workspace/MemAdaptor/wandb/offline-run-20260421_155702-adftf6mj"


)

# 方式 2 — 数组为空时，从该文件逐行读取路径（默认与脚本同目录，可改成任意绝对路径）
WANDB_OFFLINE_SYNC_LIST_FILE="${SCRIPT_DIR}/wandb_sync_paths.txt"

# 方式 3 — 若数组为空且列表文件不存在，可填写 wandb 根目录，扫描其下全部 offline-run-*；留空 "" 表示不用
WANDB_OFFLINE_SYNC_SCAN_ROOT=""

# ---------------------------------------------------------------------------

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_err() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

sync_run_dir() {
  local run_dir="$1"
  [ -d "${run_dir}" ] || return 1
  log "wandb sync ${run_dir}"
  if WANDB_MODE=online wandb sync "${run_dir}"; then
    log "完成: ${run_dir}"
  else
    log_err "失败: ${run_dir}（将下一轮重试）"
  fi
}

sync_from_list_file() {
  local path="$1"
  local any=0
  while IFS= read -r raw || [ -n "${raw}" ]; do
    line="$(trim "${raw}")"
    [ -z "${line}" ] && continue
    [[ "${line}" == \#* ]] && continue
    any=1
    if [ ! -d "${line}" ]; then
      log_err "跳过（非目录或不存在）: ${line}"
      continue
    fi
    sync_run_dir "${line}"
  done < "${path}"
  if [ "${any}" -eq 0 ]; then
    log "列表文件无有效行，跳过"
  fi
}

INTERVAL_MIN="${1:-3}"

if ! [[ "${INTERVAL_MIN}" =~ ^[0-9]+$ ]] || [ "${INTERVAL_MIN}" -lt 1 ]; then
  log_err "用法: $0 [间隔分钟数_n，默认 3]"
  log_err "示例: nohup $0 10 > wandb_sync.log 2>&1 &"
  exit 1
fi

INTERVAL_SEC=$((INTERVAL_MIN * 60))

if ! command -v wandb >/dev/null 2>&1; then
  log_err "错误: 未找到 wandb 命令，请先激活训练环境或 pip install wandb"
  exit 1
fi

USE_ARRAY=0
for _d in "${WANDB_OFFLINE_SYNC_RUN_DIRS[@]}"; do
  _t="$(trim "${_d}")"
  if [ -n "${_t}" ]; then
    USE_ARRAY=1
    break
  fi
done

SYNC_MODE=""
LIST_FILE_RESOLVED=""
SCAN_ROOT_RESOLVED=""

if [ "${USE_ARRAY}" -eq 1 ]; then
  SYNC_MODE="array"
elif [ -f "${WANDB_OFFLINE_SYNC_LIST_FILE}" ]; then
  SYNC_MODE="file"
  LIST_FILE_RESOLVED="$(cd "$(dirname "${WANDB_OFFLINE_SYNC_LIST_FILE}")" && pwd)/$(basename "${WANDB_OFFLINE_SYNC_LIST_FILE}")"
  if [ ! -r "${LIST_FILE_RESOLVED}" ]; then
    log_err "错误: 列表文件不可读: ${LIST_FILE_RESOLVED}"
    exit 1
  fi
elif [ -n "$(trim "${WANDB_OFFLINE_SYNC_SCAN_ROOT}")" ] && [ -d "${WANDB_OFFLINE_SYNC_SCAN_ROOT}" ]; then
  SYNC_MODE="scan"
  SCAN_ROOT_RESOLVED="$(cd "${WANDB_OFFLINE_SYNC_SCAN_ROOT}" && pwd)"
else
  log_err "错误: 未配置可执行的同步目标。请任选其一:"
  log_err "  - 在本脚本中填写 WANDB_OFFLINE_SYNC_RUN_DIRS，或"
  log_err "  - 创建列表文件: ${WANDB_OFFLINE_SYNC_LIST_FILE}，或"
  log_err "  - 设置 WANDB_OFFLINE_SYNC_SCAN_ROOT 为存在的 wandb 根目录"
  exit 1
fi

log "wandb 离线同步循环已启动"
log "间隔: ${INTERVAL_MIN} 分钟 (${INTERVAL_SEC} 秒)"
case "${SYNC_MODE}" in
  array)
    log "模式: 脚本内数组（${#WANDB_OFFLINE_SYNC_RUN_DIRS[@]} 项）"
    ;;
  file)
    log "模式: 列表文件 → ${LIST_FILE_RESOLVED}"
    ;;
  scan)
    log "模式: 扫描目录 → ${SCAN_ROOT_RESOLVED}/offline-run-*"
    ;;
esac
log "按 Ctrl+C 或 kill 停止"
echo ""

while true; do
  log "开始一轮同步..."

  case "${SYNC_MODE}" in
    array)
      for run_dir in "${WANDB_OFFLINE_SYNC_RUN_DIRS[@]}"; do
        run_dir="$(trim "${run_dir}")"
        [ -z "${run_dir}" ] && continue
        if [ ! -d "${run_dir}" ]; then
          log_err "跳过（非目录或不存在）: ${run_dir}"
          continue
        fi
        sync_run_dir "${run_dir}"
      done
      ;;
    file)
      sync_from_list_file "${LIST_FILE_RESOLVED}"
      ;;
    scan)
      shopt -s nullglob
      runs=( "${SCAN_ROOT_RESOLVED}"/offline-run-* )
      shopt -u nullglob
      if [ "${#runs[@]}" -eq 0 ]; then
        log "未发现 ${SCAN_ROOT_RESOLVED}/offline-run-* ，跳过"
      else
        for run_dir in "${runs[@]}"; do
          [ -d "${run_dir}" ] || continue
          sync_run_dir "${run_dir}"
        done
      fi
      ;;
  esac

  log "休眠 ${INTERVAL_MIN} 分钟"
  sleep "${INTERVAL_SEC}"
done
