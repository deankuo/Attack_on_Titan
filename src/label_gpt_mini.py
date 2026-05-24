#!/usr/bin/env python3
"""
label_gpt_mini.py — Sentence-level MD + in/out-group labeling via OpenAI chat API.

Features:
  • Synchronous calls, --workers concurrent threads (default 10).
  • Checkpoint every --batch-size rows (default 200): full CSV rewritten each time.
  • Resume: on startup scans output CSV for rows with missing scores and only
    re-runs those (first run processes everything).

Usage:
    OPENAI_API_KEY=your_key python label_gpt_mini.py [--input PATH] [--output PATH]
                                                      [--batch-size N] [--workers N]
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

MODEL            = "gpt-5-mini"
DEFAULT_INPUT    = "data/aot_labeled_filtered.csv"
DEFAULT_OUTPUT   = "data/aot_gpt_mini_labeled.csv"
MAX_TOKENS_MD    = 3000  # 8 strategies — reasoning model needs headroom
MAX_TOKENS_INOUT = 3000  # 3 indicators — reasoning model needs headroom

# Expected score columns — used to detect incomplete rows on resume
SCORE_COLS = [
    "md_moral_justification_score",
    "md_euphemistic_labeling_score",
    "md_advantageous_comparison_score",
    "md_displacement_of_responsibility_score",
    "md_diffusion_of_responsibility_score",
    "md_disregard___distortion_of_consequences_score",
    "md_dehumanization_score",
    "md_attribution_of_blame_score",
    "inout_boundary_marking_score",
    "inout_threat_framing_score",
    "inout_solidarity_appeal_score",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default=DEFAULT_INPUT)
    p.add_argument("--output",     default=DEFAULT_OUTPUT)
    p.add_argument("--batch-size", type=int, default=200,
                   help="Rows per checkpoint (default: 200)")
    p.add_argument("--workers",    type=int, default=10,
                   help="Concurrent API threads per batch (default: 10)")
    return p.parse_args()


# ── JSON parsing ──────────────────────────────────────────────────────────────

def extract_json(text):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        start = text.find("{")
        if start == -1:
            return None
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return None
        candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def flatten_md(parsed):
    if not parsed or "analysis" not in parsed:
        return {}
    out = {}
    for item in parsed["analysis"]:
        key = item.get("strategy", "unknown").lower().replace(" ", "_").replace("/", "_")
        out[f"md_{key}_score"] = item.get("score")
    return out


def flatten_inout(parsed):
    if not parsed or "analysis" not in parsed:
        return {}
    out = {}
    for item in parsed["analysis"]:
        key = item.get("strategy", "unknown").lower().replace(" ", "_")
        out[f"inout_{key}_score"] = item.get("score")
    return out


# ── API call (with retry) ─────────────────────────────────────────────────────

def call_api(client, system_prompt, sentence, max_tokens, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f'Sentence: "{sentence}"\n\nRespond ONLY with the JSON object. No explanation, no preamble.'},
                ],
                temperature=1,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                return "[EMPTY RESPONSE]"
            return content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR: {e}]"


def label_row(client, sentence, md_system, inout_system):
    md_text    = call_api(client, md_system,    sentence, MAX_TOKENS_MD)
    inout_text = call_api(client, inout_system, sentence, MAX_TOKENS_INOUT)

    md_parsed    = extract_json(md_text)
    inout_parsed = extract_json(inout_text)
    md_ok    = md_parsed    is not None
    inout_ok = inout_parsed is not None

    return {
        "md_parse_ok":    md_ok,
        "inout_parse_ok": inout_ok,
        "md_raw":         md_text    if not md_ok    else "",
        "inout_raw":      inout_text if not inout_ok else "",
        **flatten_md(md_parsed),
        **flatten_inout(inout_parsed),
    }


# ── Resume helpers ────────────────────────────────────────────────────────────

def find_todo_indices(df):
    """Return row indices where any score column is missing or both parse flags are False."""
    existing_scores = [c for c in SCORE_COLS if c in df.columns]
    if not existing_scores:
        return list(df.index)

    all_scores_present = df[existing_scores].notna().all(axis=1)
    md_ok    = df.get("md_parse_ok",    pd.Series(False, index=df.index)).fillna(False).astype(bool)
    inout_ok = df.get("inout_parse_ok", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    done = md_ok & inout_ok & all_scores_present
    return df.index[~done].tolist()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Set OPENAI_API_KEY before running.")
    client = OpenAI(api_key=api_key)

    md_system    = Path("moral_disengagement_system_prompt.txt").read_text()
    inout_system = Path("in_out_group_prompt.txt").read_text()

    df_in = pd.read_csv(args.input).reset_index(drop=True)
    n = len(df_in)
    print(f"Loaded {n} rows from {args.input}")

    # ── Load existing output or start fresh ───────────────────────────────────
    if Path(args.output).exists():
        df_out = pd.read_csv(args.output).reset_index(drop=True)
        todo = find_todo_indices(df_out)
        print(f"Resuming: {n - len(todo)} rows already done, {len(todo)} to process.")
    else:
        df_out = df_in.copy()
        todo = list(range(n))
        print(f"Fresh run: {n} rows to process.")

    if not todo:
        print("Nothing to do — all rows already labeled.")
        return

    md_fail = inout_fail = 0
    total_batches = (len(todo) + args.batch_size - 1) // args.batch_size

    for batch_num, batch_start in enumerate(range(0, len(todo), args.batch_size), 1):
        batch_indices = todo[batch_start : batch_start + args.batch_size]

        futures_map = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for idx in batch_indices:
                sentence = str(df_out.loc[idx, "sentence"])
                fut = pool.submit(label_row, client, sentence, md_system, inout_system)
                futures_map[fut] = idx

            batch_results = {}
            for fut in tqdm(
                as_completed(futures_map),
                total=len(futures_map),
                desc=f"Batch {batch_num}/{total_batches}",
                unit="row",
            ):
                idx = futures_map[fut]
                batch_results[idx] = fut.result()

        # Apply results to df_out
        for idx, result in batch_results.items():
            for col, val in result.items():
                df_out.loc[idx, col] = val
            if not result["md_parse_ok"]:
                md_fail += 1
            if not result["inout_parse_ok"]:
                inout_fail += 1

        # Checkpoint: rewrite full CSV
        df_out.to_csv(args.output, index=False)
        done_count = batch_start + len(batch_indices)
        print(f"  [{batch_num}/{total_batches}] Checkpoint: {done_count}/{len(todo)} done "
              f"| MD fails: {md_fail} | InOut fails: {inout_fail}")

    total = len(todo)
    print(f"\nDone. {len(df_out)} rows saved to {args.output}")
    print(f"Parse success — MD: {total - md_fail}/{total}, "
          f"InOut: {total - inout_fail}/{total}")


if __name__ == "__main__":
    main()
