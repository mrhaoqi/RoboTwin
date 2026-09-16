#!/bin/bash
# 推理服务端 —— 在 **lerobot 环境**里跑
# 用法: bash policy/smolvla/eval_server.sh <ckpt_path> [port] [gpu_id]
set -e
CKPT=${1:-~/train_runs/smolvla_pos_lateral/checkpoints/last/pretrained_model}
PORT=${2:-9999}
GPU=${3:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
cd "$(dirname "$0")/../.."          # 回到仓库根
PYTHONWARNINGS=ignore::UserWarning \
~/miniconda3/envs/lerobot/bin/python script/policy_model_server.py \
  --port ${PORT} \
  --config policy/smolvla/deploy_policy.yml \
  --overrides \
  --policy_name smolvla \
  --ckpt_path ${CKPT}
