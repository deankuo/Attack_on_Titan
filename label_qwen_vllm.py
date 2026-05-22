#!/usr/bin/env python3
"""
label_qwen_vllm.py — Sentence-level moral disengagement and in/out-group labeling
using a local Qwen3 model via vLLM (faster than transformers for batch inference).

Usage:
    python label_qwen_vllm.py [--input PATH] [--output-csv PATH]
                               [--model PATH] [--batch-size N] [--tp N]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "/work/10767/pengting1999/ls6/llm_models/Qwen3.6-35B-A3B"
DEFAULT_INPUT = "data/aot_labeled_filtered.csv"
DEFAULT_CSV = "data/aot_qwen_labeled.csv"

MAX_TOKENS_MD = 300     # 8 indicators × score field only
MAX_TOKENS_INOUT = 150  # 3 indicators × score field only
MAX_MODEL_LEN = 4096


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output-csv", default=DEFAULT_CSV)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=200,
                   help="Rows per vLLM submission; CSV is rewritten after each batch")
    p.add_argument("--tp", type=int, default=2,
                   help="Tensor parallel size; must divide the model's attention head count (16 for this model)")
    return p.parse_args()


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


def format_prompt(tokenizer, system_prompt, sentence):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Sentence: "{sentence}"'},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizer versions that don't support enable_thinking
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def extract_json(text):
    """Extract first JSON object, with brace-matching fallback."""
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
    result = {}
    if not parsed or "analysis" not in parsed:
        return result
    for item in parsed["analysis"]:
        key = item.get("strategy", "unknown").lower().replace(" ", "_").replace("/", "_")
        result[f"md_{key}_score"] = item.get("score")
    return result


def flatten_inout(parsed):
    result = {}
    if not parsed or "analysis" not in parsed:
        return result
    for item in parsed["analysis"]:
        key = item.get("strategy", "unknown").lower().replace(" ", "_")
        result[f"inout_{key}_score"] = item.get("score")
    return result


def main():
    args = parse_args()

    md_system = Path("moral_disengagement_system_prompt.txt").read_text()
    inout_system = Path("in_out_group_prompt.txt").read_text()

    df = pd.read_csv(args.input).reset_index(drop=True)
    total = len(df)
    print(f"Input rows: {total}", flush=True)

    print("Loading tokenizer ...", flush=True)
    tokenizer = load_tokenizer(args.model)

    print(f"Initializing vLLM (tensor_parallel_size={args.tp}) ...", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    sampling_md = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS_MD)
    sampling_inout = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS_INOUT)

    all_records: list = []
    md_fail = 0
    inout_fail = 0

    pbar = tqdm(range(0, total, args.batch_size), desc="Labeling",
                unit="batch", dynamic_ncols=True)

    for batch_start in pbar:
        batch_df = df.iloc[batch_start : batch_start + args.batch_size]
        sentences = [str(s) for s in batch_df["sentence"]]

        md_prompts = [format_prompt(tokenizer, md_system, s) for s in sentences]
        inout_prompts = [format_prompt(tokenizer, inout_system, s) for s in sentences]

        md_outputs = llm.generate(md_prompts, sampling_md)
        inout_outputs = llm.generate(inout_prompts, sampling_inout)

        for j, (_, row) in enumerate(batch_df.iterrows()):
            md_text = re.sub(r"<think>.*?</think>",
                             "", md_outputs[j].outputs[0].text, flags=re.DOTALL).strip()
            inout_text = re.sub(r"<think>.*?</think>",
                                "", inout_outputs[j].outputs[0].text, flags=re.DOTALL).strip()

            md_parsed = extract_json(md_text)
            inout_parsed = extract_json(inout_text)

            if md_parsed is None:
                md_fail += 1
            if inout_parsed is None:
                inout_fail += 1

            all_records.append({
                "season": int(row["season"]),
                "episode": int(row["episode"]),
                "title": row.get("title", ""),
                "airdate": row.get("airdate", ""),
                "character": row.get("character", ""),
                "sentence": row["sentence"],
                "md_parse_ok": md_parsed is not None,
                "inout_parse_ok": inout_parsed is not None,
                "md_raw": md_text if md_parsed is None else "",
                "inout_raw": inout_text if inout_parsed is None else "",
                **flatten_md(md_parsed),
                **flatten_inout(inout_parsed),
            })

        # Checkpoint: rewrite CSV after every batch
        pd.DataFrame(all_records).to_csv(args.output_csv, index=False)

        pbar.set_postfix(
            rows=batch_start + len(batch_df),
            md_fail=md_fail,
            inout_fail=inout_fail,
        )

    # Final save + summary
    out_df = pd.DataFrame(all_records)
    out_df.to_csv(args.output_csv, index=False)
    n = len(out_df)
    print(f"Saved {n} rows to {args.output_csv}", flush=True)
    print(f"Parse success — MD: {n - md_fail}/{n}, InOut: {n - inout_fail}/{n}", flush=True)


if __name__ == "__main__":
    main()
