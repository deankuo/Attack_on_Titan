#!/bin/bash
#------------------------------------------------------------
# SLURM job script for TACC Lonestar6
# Runs label_qwen_vllm.py with Qwen2.5-32B-Instruct on 3× A100 GPUs
# using tensor parallelism (--tp 3) — one process, model sharded
# across all three GPUs (~64 GB weights; requires multi-GPU).
#
# Submit with:  sbatch run_label.sh
# Fill in <FILL_IN_ALLOCATION> with your TACC allocation account.
#------------------------------------------------------------

#SBATCH -J qwen_label
#SBATCH -o logs/qwen_label_%j.out
#SBATCH -e logs/qwen_label_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a100:3               # all 3 A100s for tensor parallelism
#SBATCH -p gpu-a100
#SBATCH -A <FILL_IN_ALLOCATION>

PROJECT_DIR="$WORK/SCISS"

source ~/.bashrc
conda activate vllm_env

cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

mkdir -p logs data

echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Start time  : $(date)"
echo "Working dir : $(pwd)"
echo "=============================="

# V1 engine forks an EngineCore subprocess; CUDA cannot be re-initialized
# in a forked process → use spawn.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="Qwen/Qwen2.5-32B-Instruct"

# Qwen2.5-32B has 40 attention heads — not divisible by 3.
# --tp 2 uses 2 GPUs (80 GB), fitting the ~64 GB model; 3rd GPU is idle.
python label_qwen_vllm.py \
    --input      data/aot_labeled_filtered.csv \
    --output-csv data/aot_qwen_labeled.csv \
    --model      $MODEL \
    --tp         2 \
    --batch-size 200

echo "=============================="
echo "End time : $(date)"
echo "=============================="
