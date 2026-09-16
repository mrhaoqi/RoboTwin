#!/bin/bash
# 仿真客户端 —— 在 **RoboTwin 环境**里跑(需先启动 eval_server.sh)
# 用法: bash policy/smolvla/eval_client.sh <task_name> <task_config> <ckpt_setting> <seed> [port] [gpu_id]
set -e
TASK=${1}
TASK_CONFIG=${2}
CKPT_SETTING=${3:-last}
SEED=${4:-0}
PORT=${5:-9999}
GPU=${6:-0}
export CUDA_VISIBLE_DEVICES=${GPU}
cd "$(dirname "$0")/../.."
PYTHONWARNINGS=ignore::UserWarning \
~/miniconda3/envs/RoboTwin/bin/python script/eval_policy_client.py \
  --port ${PORT} \
  --config policy/smolvla/deploy_policy.yml \
  --overrides \
  --task_name ${TASK} \
  --task_config ${TASK_CONFIG} \
  --ckpt_setting ${CKPT_SETTING} \
  --seed ${SEED} \
  --policy_name smolvla
