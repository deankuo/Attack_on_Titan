#!/usr/bin/env python3
"""
label_qwen_vllm.py — Sentence-level moral disengagement and in/out-group labeling
using a local Qwen3 model via vLLM.
"""

import os

# vLLM V1 engine forks a subprocess for EngineCore; CUDA cannot be
# re-initialized in a forked process → use spawn start method.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# transformers 5.8.1 added 'aimv2' as a built-in config; vLLM 0.9.0's
# ovis.py tries to re-register it, raising ValueError.  Patch the
# CONFIG_MAPPING.register call to be idempotent before vLLM is imported.
import transformers.models.auto.configuration_auto as _ca
_orig_register = _ca.CONFIG_MAPPING.register

def _safe_register(key, value, exist_ok=False):
    _orig_register(key, value, exist_ok=True)

_ca.CONFIG_MAPPING.register = _safe_register

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

# vLLM 0.8.5 requires all_special_tokens_extended, added in transformers 4.38.
# Patch the base class if missing so older envs work unchanged.
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )

DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_INPUT = "data/aot_labeled_filtered.csv"
DEFAULT_CSV = "data/aot_qwen_labeled.csv"

# MD system prompt is ~2000 tokens; MAX_MODEL_LEN must cover prompt + output.
# 8192 gives ~6000 tokens of headroom for generation after the prompt.
MAX_TOKENS_MD = 2000    # 8 indicators × ~150 tokens each + structure
MAX_TOKENS_INOUT = 600  # 3 indicators
MAX_MODEL_LEN = 16000

RETRY_MAX_TOKENS_MD = 3000   # retry pass — max headroom for stubborn truncations
RETRY_MAX_TOKENS_INOUT = 1200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output-csv", default=DEFAULT_CSV)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=200,
                   help="Rows per vLLM submission; CSV is rewritten after each batch")
    p.add_argument("--tp", type=int, default=2,
                   help="Tensor parallel size (Qwen2.5-32B has 40 heads; valid: 1,2,4,5,…)")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="Process shard I of N rows (0-indexed). "
                        "Output CSV is auto-suffixed _shardI. e.g. --shard 0/3")
    p.add_argument("--retry", action="store_true",
                   help="Re-run only failed rows (md_parse_ok=False or inout_parse_ok=False) "
                        "from an existing --output-csv, then update it in place.")
    p.add_argument("--retry-max-tokens-md", type=int, default=RETRY_MAX_TOKENS_MD)
    p.add_argument("--retry-max-tokens-inout", type=int, default=RETRY_MAX_TOKENS_INOUT)
    return p.parse_args()


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)


def format_prompt(tokenizer, system_prompt, sentence):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Sentence: "{sentence}"'},
    ]
    # Disable Qwen3 thinking mode: <think> blocks consume max_tokens budget
    # before the JSON is written, causing near-total parse failures.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizer versions don't support enable_thinking
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


# Keyword → canonical column name.  First match wins, so put more specific
# terms before shorter ones that could appear in multiple strategy names.
_MD_KEYWORDS = [
    ("moral_justification",    "md_moral_justification_score"),
    ("euphemistic",            "md_euphemistic_labeling_score"),
    ("advantageous",           "md_advantageous_comparison_score"),
    ("displacement",           "md_displacement_of_responsibility_score"),
    ("diffusion",              "md_diffusion_of_responsibility_score"),
    ("disregard",              "md_disregard_distortion_of_consequences_score"),
    ("distortion",             "md_disregard_distortion_of_consequences_score"),
    ("dehumanization",         "md_dehumanization_score"),
    ("attribution",            "md_attribution_of_blame_score"),
]

_INOUT_KEYWORDS = [
    ("boundary",   "inout_boundary_marking_score"),
    ("threat",     "inout_threat_framing_score"),
    ("solidarity", "inout_solidarity_appeal_score"),
]


def _resolve_col(strategy: str, keyword_map: list, prefix: str) -> str:
    slug = strategy.lower()
    for keyword, col in keyword_map:
        if keyword in slug:
            return col
    # Fallback for truly unexpected strategy names
    safe = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return f"{prefix}{safe}_score"


def flatten_md(parsed):
    result = {}
    if not parsed or "analysis" not in parsed:
        return result
    for item in parsed["analysis"]:
        if not isinstance(item, dict):
            continue
        col = _resolve_col(str(item.get("strategy", "")), _MD_KEYWORDS, "md_")
        result[col] = item.get("score")
    return result


def flatten_inout(parsed):
    result = {}
    if not parsed or "analysis" not in parsed:
        return result
    for item in parsed["analysis"]:
        if not isinstance(item, dict):
            continue
        col = _resolve_col(str(item.get("strategy", "")), _INOUT_KEYWORDS, "inout_")
        result[col] = item.get("score")
    return result


def _run_retry_pass(label, df, indices, prompts_fn, sampling, flatten_fn,
                    ok_col, raw_col, score_prefix, llm, batch_size):
    """Re-run one task (MD or InOut) for a subset of row indices; update df in place."""
    fixed = 0
    pbar = tqdm(range(0, len(indices), batch_size), desc=f"Retry {label}",
                unit="batch", dynamic_ncols=True)
    for start in pbar:
        chunk_idx = indices[start : start + batch_size]
        sentences = [str(df.loc[i, "sentence"]) for i in chunk_idx]
        outputs = llm.generate(prompts_fn(sentences), sampling)

        for row_idx, output in zip(chunk_idx, outputs):
            text = re.sub(r"<think>.*?</think>",
                          "", output.outputs[0].text, flags=re.DOTALL).strip()
            parsed = extract_json(text)
            if parsed is not None:
                flat = flatten_fn(parsed)
                df.loc[row_idx, ok_col] = True
                df.loc[row_idx, raw_col] = ""
                for col, val in flat.items():
                    df.loc[row_idx, col] = val
                fixed += 1
            else:
                df.loc[row_idx, raw_col] = text  # keep freshest raw for debugging
        pbar.set_postfix(fixed=fixed)
    return fixed


def retry_failed(args, llm, tokenizer, md_system, inout_system):
    out_path = args.output_csv
    if not Path(out_path).exists():
        raise FileNotFoundError(f"Output CSV not found: {out_path} — run without --retry first.")

    df = pd.read_csv(out_path)

    # Ensure bool dtype (CSV round-trips may read as object)
    for col in ("md_parse_ok", "inout_parse_ok"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(False)

    md_fail_idx   = df.index[~df["md_parse_ok"]].tolist()
    inout_fail_idx = df.index[~df["inout_parse_ok"]].tolist()

    print(f"Rows needing retry — MD: {len(md_fail_idx)}, InOut: {len(inout_fail_idx)}", flush=True)

    sampling_md    = SamplingParams(temperature=0.0, max_tokens=args.retry_max_tokens_md)
    sampling_inout = SamplingParams(temperature=0.0, max_tokens=args.retry_max_tokens_inout)

    if md_fail_idx:
        def md_prompts(sentences):
            return [format_prompt(tokenizer, md_system, s) for s in sentences]
        fixed = _run_retry_pass(
            "MD", df, md_fail_idx, md_prompts, sampling_md,
            flatten_md, "md_parse_ok", "md_raw", "md_", llm, args.batch_size,
        )
        print(f"MD retry: fixed {fixed}/{len(md_fail_idx)}", flush=True)

    if inout_fail_idx:
        def inout_prompts(sentences):
            return [format_prompt(tokenizer, inout_system, s) for s in sentences]
        fixed = _run_retry_pass(
            "InOut", df, inout_fail_idx, inout_prompts, sampling_inout,
            flatten_inout, "inout_parse_ok", "inout_raw", "inout_", llm, args.batch_size,
        )
        print(f"InOut retry: fixed {fixed}/{len(inout_fail_idx)}", flush=True)

    df.to_csv(out_path, index=False)
    remaining_md    = (~df["md_parse_ok"]).sum()
    remaining_inout = (~df["inout_parse_ok"]).sum()
    print(f"Saved {out_path}. Still failing — MD: {remaining_md}, InOut: {remaining_inout}", flush=True)


def main():
    args = parse_args()

    md_system = Path("moral_disengagement_system_prompt.txt").read_text()
    inout_system = Path("in_out_group_prompt.txt").read_text()

    if not args.retry:
        df = pd.read_csv(args.input).reset_index(drop=True)

        # --- sharding ---
        if args.shard:
            shard_idx, n_shards = (int(x) for x in args.shard.split("/"))
            chunk = math.ceil(len(df) / n_shards)
            df = df.iloc[shard_idx * chunk : (shard_idx + 1) * chunk].reset_index(drop=True)
            p_out = Path(args.output_csv)
            args.output_csv = str(p_out.parent / f"{p_out.stem}_shard{shard_idx}{p_out.suffix}")
            print(f"Shard {shard_idx}/{n_shards}: {len(df)} rows → {args.output_csv}", flush=True)

        total = len(df)
        print(f"Total rows to label: {total}", flush=True)

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
        enforce_eager=True,
    )

    if args.retry:
        retry_failed(args, llm, tokenizer, md_system, inout_system)
        return

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
