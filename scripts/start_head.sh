#!/bin/bash
#SBATCH --job-name=evolver-1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=DataFrontier_Explore
#SBATCH --quotatype=reserved


### 系统变量, 请按需修改！！！
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
source ~/switch_cuda.sh 12.4
nvcc --version
eval "$(/mnt/petrelfs/wurong/miniconda3/bin/conda shell.bash hook)"
conda activate verl-agent # 记得改成自己的
conda info


export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_MODELSCOPE="0"
export WANDB_MODE="offline"
export SWANLAB_MODE="offline"
export MKL_SERVICE_FORCE_INTEL=1
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

set -x
ENGINE=${1:-vllm}
###

set -xe
# 自定义路径，过长或者机器上已有此目录会报错
RAYLOG="/tmp/wr-ray"

# 获取节点列表
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# 获取节点ip
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 如果机器开启IPv6会执行 确保获取IPv4地址
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=8249 # 默认为6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Slurm 上 --gpus-per-task 不一定会设置 SLURM_GPUS_PER_TASK；若把空串传给 Ray，可能注册 0 张 GPU，提交的任务会一直 PENDING。
if [ -n "${SLURM_GPUS_PER_TASK:-}" ]; then
  RAY_NUM_GPUS="${SLURM_GPUS_PER_TASK}"
else
  RAY_NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
RAY_NUM_GPUS="$(echo "${RAY_NUM_GPUS}" | tr -d '[:space:]')"
if ! [[ "${RAY_NUM_GPUS}" =~ ^[0-9]+$ ]] || [ "${RAY_NUM_GPUS}" -eq 0 ]; then
  RAY_NUM_GPUS=8
fi
echo "Ray --num-gpus=${RAY_NUM_GPUS} (SLURM_GPUS_PER_TASK=${SLURM_GPUS_PER_TASK:-<unset>}, nvidia-smi -L count used if unset)"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --temp-dir=$RAYLOG \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${RAY_NUM_GPUS}" --block --dashboard-host=0.0.0.0 --disable-usage-stats

sleep infinity