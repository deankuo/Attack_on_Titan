#!/bin/bash
# ============================================================================
# run_classify.sh — SLURM job script for TACC
#
# BEFORE SUBMITTING:
#   1. Set --account to your TACC allocation
#   2. Set --partition to match your system:
#        Frontera  → rtx  (or rtx-dev for testing, max 2h)
#        Vista     → gpu-a100-dev  (dev) or gpu-a100  (production)
#        Lonestar6 → gpu-a100  (or gpu-a100-dev)
#   3. Adjust --time if needed (transcripts ~20 min, larger comment sets longer)
#   4. Run the download step on the LOGIN NODE first (see below)
#
# USAGE:
#   # Step 1 — download models on login node (one-time, ~2 GB):
#   export WORK=/work/<username>/<allocation>   # or echo $WORK
#   python classify.py --download-only --models-dir $WORK/hf_cache
#
#   # Step 2 — submit job:
#   sbatch run_classify.sh
#
#   # Step 3 — monitor:
#   squeue -u $USER
#   tail -f logs/classify_<jobid>.out
# ============================================================================

#SBATCH --job-name=aot_classify
#SBATCH --output=logs/classify_%j.out
#SBATCH --error=logs/classify_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100-dev          # <-- change for your system (see above)
#SBATCH --account=YOUR_ALLOCATION         # <-- change to your allocation
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=lieutenantl626@gmail.com

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
source llm/bin/activate

# Load CUDA module — check available versions with: module avail cuda
# module load cuda/12.2   # Vista / LS6
# module load cuda/11.3   # Frontera

# Keep models in $WORK (large, persistent) — never $HOME
export HF_HOME=$WORK/hf_cache
mkdir -p "$HF_HOME" logs

echo "===== Job info ====="
echo "Host     : $(hostname)"
echo "Start    : $(date)"
echo "HF_HOME  : $HF_HOME"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "===================="

# ── AoT transcript dataset ───────────────────────────────────────────────────
python classify.py \
    --input      data/aot_labeled.csv \
    --output     data/aot_classified.csv \
    --text-col   sentence \
    --task       both \
    --models-dir "$HF_HOME" \
    --batch-size 128 \
    --device     0 

# ── User comments dataset (uncomment and adjust when ready) ──────────────────
# python classify.py \
#     --input      data/comments.csv \
#     --output     data/comments_classified.csv \
#     --text-col   comment \           # change to your actual column name
#     --task       both \
#     --models-dir "$HF_HOME" \
#     --batch-size 128 \
#     --device     0 \
#     --resume

echo "Done : $(date)"
