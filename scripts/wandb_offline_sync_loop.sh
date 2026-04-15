#!/usr/bin/env bash
# 周期性将本地 W&B 离线 run 目录同步到云端。
#
# 用法:
#   chmod +x scripts/wandb_offline_sync_loop.sh
#   export WANDB_API_KEY=...   # 与训练时一致
#   nohup ./scripts/wandb_offline_sync_loop.sh 10 ./wandb > wandb_sync.log 2>&1 &
#
# 参数:
#   $1  间隔分钟数 n（默认 10）
#   $2  wandb 根目录（默认 ./wandb，内含 offline-run-*）
#
# 说明:
#   - 训练侧保持 WANDB_MODE=offline 即可；本脚本对每个 offline-run-* 调用 wandb sync，并临时设 WANDB_MODE=online。
#   - 需当前环境已安装 wandb 且能访问 W&B 服务。
#   - 同一 run 多次 sync 一般可接受；若出现重复 run，可改由 wandb 网页端处理或仅同步尚未上传的目录。

set -u

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_err() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

INTERVAL_MIN="${1:-10}"
WANDB_ROOT="${2:-./wandb}"

if ! [[ "${INTERVAL_MIN}" =~ ^[0-9]+$ ]] || [ "${INTERVAL_MIN}" -lt 1 ]; then
  log_err "用法: $0 <间隔分钟数_n> [wandb目录]"
  log_err "示例: nohup $0 15 ./wandb > wandb_sync.log 2>&1 &"
  exit 1
fi

if [ ! -d "${WANDB_ROOT}" ]; then
  log_err "错误: wandb 目录不存在: ${WANDB_ROOT}"
  exit 1
fi

WANDB_ROOT="$(cd "${WANDB_ROOT}" && pwd)"
INTERVAL_SEC=$((INTERVAL_MIN * 60))

if ! command -v wandb >/dev/null 2>&1; then
  log_err "错误: 未找到 wandb 命令，请先激活训练环境或 pip install wandb"
  exit 1
fi

log "wandb 离线同步循环已启动"
log "间隔: ${INTERVAL_MIN} 分钟 (${INTERVAL_SEC} 秒)"
log "目录: ${WANDB_ROOT}"
log "按 Ctrl+C 或 kill 停止"
echo ""

while true; do
  log "开始一轮同步..."

  shopt -s nullglob
  runs=( "${WANDB_ROOT}"/offline-run-* )
  shopt -u nullglob

  if [ "${#runs[@]}" -eq 0 ]; then
    log "未发现 ${WANDB_ROOT}/offline-run-* ，跳过"
  else
    for run_dir in "${runs[@]}"; do
      [ -d "${run_dir}" ] || continue
      log "wandb sync ${run_dir}"
      if WANDB_MODE=online wandb sync "${run_dir}"; then
        log "完成: ${run_dir}"
      else
        log_err "失败: ${run_dir}（将下一轮重试）"
      fi
    done
  fi

  log "休眠 ${INTERVAL_MIN} 分钟"
  sleep "${INTERVAL_SEC}"
done
