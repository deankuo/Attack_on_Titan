#!/bin/bash
#------------------------------------------------------------
# SLURM job script for TACC Lonestar6
# Runs label_qwen_vllm.py with Qwen3 on 3× A100 GPUs via vLLM
#
# Submit with:  sbatch run_label.sh
#
# Fill in <FILL_IN_ALLOCATION> with your TACC allocation name
# (e.g., the output of `myprojectlist` or `accounts` on LS6).
#------------------------------------------------------------

#SBATCH -J qwen_label                   # job name
#SBATCH -o logs/qwen_label_%j.out       # stdout  (%j = job ID)
#SBATCH -e logs/qwen_label_%j.err       # stderr
#SBATCH -N 1                            # 1 node
#SBATCH -n 1                            # 1 MPI task
#SBATCH --cpus-per-task=16              # CPU cores for data loading / tokenization
#SBATCH --mem=128G                      # host RAM
#SBATCH -t 48:00:00                     # wall time limit
#SBATCH --gres=gpu:a100:2               # 2× A100 40GB (tp=2; 16 heads not divisible by 3)
#SBATCH -p gpu-a100                     # partition
#SBATCH -A <FILL_IN_ALLOCATION>         # <-- replace with your allocation account

# ---- environment setup ----------------------------------------
source ~/.bashrc
source llm/bin/activate

cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

mkdir -p logs data

echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Start time  : $(date)"
echo "Working dir : $(pwd)"
echo "=============================="

# ---- run vLLM labeling script ---------------------------------
# --tp 3  → tensor parallel across all 3 A100s (120 GB VRAM total)
# --chunk-size 200 → write JSONL checkpoint every 200 rows
python label_qwen_vllm.py \
    --input        data/aot_labeled_filtered.csv \
    --output-jsonl data/aot_qwen_labels.jsonl \
    --output-csv   data/aot_qwen_labeled.csv \
    --model        /work/10767/pengting1999/ls6/llm_models/Qwen3.6-35B-A3B \
    --tp           2 \
    --chunk-size   200

echo "=============================="
echo "End time : $(date)"
echo "=============================="
