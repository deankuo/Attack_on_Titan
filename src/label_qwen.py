#!/usr/bin/env python3
"""
label_qwen.py — Sentence-level moral disengagement and in/out-group labeling
using a local Qwen3 model via HuggingFace transformers.

Usage:
    python label_qwen.py [--input PATH] [--output-jsonl PATH] [--output-csv PATH]
                         [--model PATH] [--batch-size N] [--no-think]
"""

import argparse
import json
import os
import re
from pathlib import Path

from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "/work/10767/pengting1999/ls6/llm_models/Qwen3.6-35B-A3B"
DEFAULT_INPUT = "data/aot_labeled_filtered.csv"
DEFAULT_JSONL = "data/aot_qwen_labels.jsonl"
DEFAULT_CSV = "data/aot_qwen_labeled.csv"

MAX_NEW_TOKENS_MD = 300    # 8 indicators × score only
MAX_NEW_TOKENS_INOUT = 150  # 3 indicators × score only
MAX_INPUT_LENGTH = 4096     # covers long system prompts + sentence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output-jsonl", default=DEFAULT_JSONL)
    p.add_argument("--output-csv", default=DEFAULT_CSV)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--no-think", action="store_true",
                   help="Append /no_think to system prompts (faster, no internal CoT)")
    return p.parse_args()


def load_model(model_path):
    print(f"Loading tokenizer from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this may take several minutes) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Device map: {getattr(model, 'hf_device_map', 'single device')}", flush=True)
    return tokenizer, model


def first_device(model):
    return next(model.parameters()).device


def build_messages(system_prompt, sentence):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Sentence: "{sentence}"'},
    ]


def batch_generate(tokenizer, model, messages_list, max_new_tokens):
    texts = []
    for msgs in messages_list:
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            # Fallback if chat template unavailable
            sys_msg = msgs[0]["content"]
            usr_msg = msgs[1]["content"]
            text = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{usr_msg}<|im_end|>\n<|im_start|>assistant\n"
        texts.append(text)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(first_device(model))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    responses = []
    input_len = inputs["input_ids"].shape[1]
    for out in outputs:
        new_tokens = out[input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Strip internal <think>...</think> block if present
        decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
        responses.append(decoded)

    return responses


def extract_json(text):
    """Extract first JSON object from model output, with fallback brace-matching."""
    # Try to find ```json ... ``` block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        # Find outermost { ... }
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = -1
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
        # Try to repair trivial issues: trailing commas
        candidate_fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate_fixed)
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


def retry_generate(tokenizer, model, messages, max_new_tokens, attempts=2):
    """Single-item generation with retry on parse failure."""
    for attempt in range(attempts):
        responses = batch_generate(tokenizer, model, [messages], max_new_tokens)
        parsed = extract_json(responses[0])
        if parsed is not None:
            return parsed, responses[0]
        # On retry, nudge temperature slightly higher to break repetition
        # (we only have one shot per attempt in this helper)
    return None, responses[0]


def load_done_indices(jsonl_path):
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["_row_idx"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main():
    args = parse_args()

    md_system = Path("moral_disengagement_system_prompt.txt").read_text()
    inout_system = Path("in_out_group_prompt.txt").read_text()

    if args.no_think:
        md_system = md_system + "\n/no_think"
        inout_system = inout_system + "\n/no_think"

    df = pd.read_csv(args.input).reset_index(drop=True)
    print(f"Input rows: {len(df)}", flush=True)

    done_indices = load_done_indices(args.output_jsonl)
    print(f"Already labeled: {len(done_indices)}", flush=True)

    pending = [(i, row) for i, row in df.iterrows() if i not in done_indices]
    print(f"Remaining: {len(pending)}", flush=True)

    if not pending:
        print("Nothing to do. Converting JSONL to CSV.", flush=True)
    else:
        tokenizer, model = load_model(args.model)

        total = len(pending)
        md_fail = 0
        inout_fail = 0

        with open(args.output_jsonl, "a", buffering=1) as f_out:
            pbar = tqdm(range(0, total, args.batch_size), desc="Labeling",
                        unit="batch", dynamic_ncols=True)
            for batch_start in pbar:
                batch = pending[batch_start : batch_start + args.batch_size]
                indices = [i for i, _ in batch]
                rows = [r for _, r in batch]
                sentences = [str(r["sentence"]) for r in rows]

                # --- Moral disengagement ---
                md_msgs_list = [build_messages(md_system, s) for s in sentences]
                md_responses = batch_generate(
                    tokenizer, model, md_msgs_list, MAX_NEW_TOKENS_MD
                )

                # --- In/out-group ---
                inout_msgs_list = [build_messages(inout_system, s) for s in sentences]
                inout_responses = batch_generate(
                    tokenizer, model, inout_msgs_list, MAX_NEW_TOKENS_INOUT
                )

                for j, (idx, row) in enumerate(zip(indices, rows)):
                    md_parsed = extract_json(md_responses[j])
                    inout_parsed = extract_json(inout_responses[j])

                    # Single-item retry if parse failed
                    if md_parsed is None:
                        md_parsed, md_responses[j] = retry_generate(
                            tokenizer, model,
                            build_messages(md_system, sentences[j]),
                            MAX_NEW_TOKENS_MD,
                        )
                    if inout_parsed is None:
                        inout_parsed, inout_responses[j] = retry_generate(
                            tokenizer, model,
                            build_messages(inout_system, sentences[j]),
                            MAX_NEW_TOKENS_INOUT,
                        )

                    record = {
                        "_row_idx": idx,
                        "season": int(row["season"]),
                        "episode": int(row["episode"]),
                        "title": row.get("title", ""),
                        "airdate": row.get("airdate", ""),
                        "character": row.get("character", ""),
                        "sentence": row["sentence"],
                        "md_parse_ok": md_parsed is not None,
                        "inout_parse_ok": inout_parsed is not None,
                        "md_raw": md_responses[j] if md_parsed is None else "",
                        "inout_raw": inout_responses[j] if inout_parsed is None else "",
                        **flatten_md(md_parsed),
                        **flatten_inout(inout_parsed),
                    }
                    if md_parsed is None:
                        md_fail += 1
                    if inout_parsed is None:
                        inout_fail += 1
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                pbar.set_postfix(
                    rows=batch_start + len(batch),
                    md_fail=md_fail,
                    inout_fail=inout_fail,
                )

    # Convert JSONL → CSV
    print("Converting JSONL to CSV ...", flush=True)
    records = []
    with open(args.output_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if records:
        records.sort(key=lambda r: r.get("_row_idx", 0))
        out_df = pd.DataFrame(records)
        # Drop internal fields from the final CSV
        out_df = out_df.drop(columns=["_row_idx"], errors="ignore")
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(out_df)} rows to {args.output_csv}", flush=True)

        parse_ok_md = out_df.get("md_parse_ok", pd.Series(dtype=bool)).sum()
        parse_ok_inout = out_df.get("inout_parse_ok", pd.Series(dtype=bool)).sum()
        n = len(out_df)
        print(f"Parse success — MD: {parse_ok_md}/{n}, InOut: {parse_ok_inout}/{n}", flush=True)
    else:
        print("No records to write.", flush=True)


if __name__ == "__main__":
    main()
