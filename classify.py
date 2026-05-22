#!/usr/bin/env python3
"""
classify.py — Emotion and valence (sentiment) classification.

Models:
  emotion:   j-hartmann/emotion-english-distilroberta-base
             labels: anger, disgust, fear, joy, neutral, sadness, surprise
  sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
             labels: negative, neutral, positive

Examples:
  # AoT transcript, both tasks (default text column: sentence)
  python classify.py --input data/aot_labeled.csv --output data/aot_classified.csv

  # User comments, different text column, emotion only
  python classify.py --input data/comments.csv --text-col comment \\
      --task emotion --output data/comments_classified.csv

  # Pre-download models to TACC $WORK (run on login node before submitting job)
  python classify.py --download-only --models-dir $WORK/hf_cache

  # Resume an interrupted job
  python classify.py --input data/aot_labeled.csv --output data/aot_classified.csv \\
      --resume
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm


EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def load_pipelines(tasks: list[str], models_dir: Path, device: int) -> dict:
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)

    pipes = {}
    if "emotion" in tasks:
        print(f"Loading {EMOTION_MODEL_ID} ...", flush=True)
        pipes["emotion"] = pipeline(
            "text-classification",
            model=EMOTION_MODEL_ID,
            top_k=None,
            device=device,
        )
    if "sentiment" in tasks:
        print(f"Loading {SENTIMENT_MODEL_ID} ...", flush=True)
        pipes["sentiment"] = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL_ID,
            top_k=None,
            device=device,
        )
    return pipes


def run_pipe(pipe, texts: list[str], batch_size: int, desc: str) -> list:
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        out = pipe(batch, truncation=True, max_length=512)
        results.extend(out)
    return results


def flatten_emotion(results: list) -> list[dict]:
    rows = []
    for result in results:
        scores = {item["label"]: item["score"] for item in result}
        top = max(result, key=lambda x: x["score"])["label"]
        row = {"emotion_label": top}
        for label in EMOTION_LABELS:
            row[f"emotion_{label}"] = round(scores.get(label, 0.0), 6)
        rows.append(row)
    return rows


def flatten_sentiment(results: list) -> list[dict]:
    rows = []
    for result in results:
        # Model returns lowercase labels with the "latest" version
        scores = {item["label"].lower(): item["score"] for item in result}
        top = max(result, key=lambda x: x["score"])["label"].lower()
        row = {"sentiment_label": top}
        for label in SENTIMENT_LABELS:
            row[f"sentiment_{label}"] = round(scores.get(label, 0.0), 6)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Emotion/sentiment classification for transcript or comment datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", help="Input CSV path (required unless --download-only)")
    parser.add_argument("--output", help="Output CSV path (required unless --download-only)")
    parser.add_argument(
        "--text-col", default="sentence",
        help="Column containing text to classify (default: sentence)",
    )
    parser.add_argument(
        "--task", choices=["emotion", "sentiment", "both"], default="both",
        help="Which task(s) to run (default: both)",
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        help="Model cache directory — set to $WORK/hf_cache on TACC (default: $HF_HOME)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size; increase to 128-256 with a GPU (default: 64)",
    )
    parser.add_argument(
        "--device", type=int,
        default=0 if torch.cuda.is_available() else -1,
        help="Device index: 0+ for GPU, -1 for CPU (default: auto-detect)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If output already exists, skip rows already classified and append new results",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download models to --models-dir then exit (run on login node before job submission)",
    )
    args = parser.parse_args()

    tasks = ["emotion", "sentiment"] if args.task == "both" else [args.task]
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Download-only mode ──────────────────────────────────────────────────
    if args.download_only:
        load_pipelines(tasks, models_dir, device=-1)
        print(f"Models cached at {models_dir}")
        return

    # ── Validate required args ──────────────────────────────────────────────
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --download-only is set")

    # ── Load data ───────────────────────────────────────────────────────────
    print(f"Reading {args.input}", flush=True)
    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        sys.exit(
            f"Column '{args.text_col}' not found in input. "
            f"Available columns: {list(df.columns)}"
        )

    # ── Resume: skip already-classified rows ────────────────────────────────
    start_idx = 0
    if args.resume and Path(args.output).exists():
        done_df = pd.read_csv(args.output)
        start_idx = len(done_df)
        print(f"Resuming: {start_idx} rows already done, {len(df) - start_idx} remaining", flush=True)

    work_df = df.iloc[start_idx:].reset_index(drop=True).copy()
    texts = work_df[args.text_col].fillna("").tolist()

    if not texts:
        print("Nothing to classify — output is already complete.")
        return

    device_label = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    print(
        f"Rows: {len(texts)} | Device: {device_label} | Batch size: {args.batch_size}",
        flush=True,
    )

    # ── Load models ─────────────────────────────────────────────────────────
    pipes = load_pipelines(tasks, models_dir, args.device)

    # ── Emotion ─────────────────────────────────────────────────────────────
    if "emotion" in pipes:
        print("Running emotion classification...", flush=True)
        results = run_pipe(pipes["emotion"], texts, args.batch_size, "emotion")
        emotion_df = pd.DataFrame(flatten_emotion(results))
        for col in emotion_df.columns:
            work_df[col] = emotion_df[col].values

    # ── Sentiment ───────────────────────────────────────────────────────────
    if "sentiment" in pipes:
        print("Running sentiment/valence classification...", flush=True)
        results = run_pipe(pipes["sentiment"], texts, args.batch_size, "sentiment")
        sentiment_df = pd.DataFrame(flatten_sentiment(results))
        for col in sentiment_df.columns:
            work_df[col] = sentiment_df[col].values

    # ── Merge resume prefix and save ────────────────────────────────────────
    if args.resume and start_idx > 0:
        done_df = pd.read_csv(args.output)
        work_df = pd.concat([done_df, work_df], ignore_index=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    work_df.to_csv(args.output, index=False)
    print(f"Saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
