#!/usr/bin/env python3
"""Label AoT transcript lines with character names using the Gemini Batch API.

Install dependency first:
    pip install google-genai

Usage:
    GEMINI_API_KEY=your_key python label_gemini.py

The script saves the batch job name to JOB_FILE on submit. If it's interrupted
during polling, re-run it — it will resume the existing job instead of
resubmitting.
"""

import os
import json
import time
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE  = "./data/aot.csv"
OUTPUT_FILE = "./data/aot_labeled.csv"
JOB_FILE    = "./data/batch_job_name.txt"
MODEL       = "gemini-2.5-pro"
POLL_INTERVAL = 60  # seconds between status checks

# fmt: off
AOT_CHARACTERS = [
    # Core Survey Corps
    "Eren Yeager", "Mikasa Ackerman", "Armin Arlert", "Levi Ackerman",
    "Hange Zoë", "Erwin Smith", "Jean Kirstein", "Connie Springer",
    "Sasha Blouse", "Historia Reiss", "Ymir", "Moblit Berner",
    "Mike Zacharias", "Floch Forster", "Isabel Magnolia", "Furlan Church",
    "Petra Ral", "Oluo Bozado", "Eld Jinn", "Gunther Schultz",
    "Nanaba", "Gelgar", "Ilse Langnar", "Dita Ness", "Luke Siss",
    "Dieter Ness", "Daz", "Samuel Linke-Jackson", "Louise",

    # Garrison / Military Police
    "Hannes", "Dot Pixis", "Keith Shadis", "Nile Dok", "Kenny Ackerman",
    "Boris Feulner", "Rashad",

    # Warriors / Marleyans
    "Reiner Braun", "Bertholdt Hoover", "Annie Leonhart",
    "Zeke Yeager", "Pieck Finger", "Porco Galliard", "Gabi Braun", "Falco Grice",
    "Colt Grice", "Theo Magath", "Willy Tybur",

    # Yeager family / supporting
    "Grisha Yeager", "Carla Yeager", "Dina Fritz",
    "Fay Yeager", "Zeke's grandmother",

    # Royal family / Church / Walls
    "Rod Reiss", "Uri Reiss", "Frieda Reiss", "Pastor Nick",
    "Abel", "Flegel Reeves", "Dimo Reeves",

    # Marley allies / Anti-Marleyan Volunteers
    "Yelena", "Onyankopon",

    # Azumabito / Eastern allies
    "Kiyomi Azumabito",

    # Titan shifter predecessors / flashback
    "Tom Ksaver", "Ymir Fritz", "Marcel Galliard", "Bertholdt's father",

    # Paradis political / military leadership
    "Zachary Daz", "Hitch Dreyse", "Marlowe Freudenberg",
    "Sandra", "Gordon", "Thomas",

    # 104th Cadet Corps peers
    "Marco Bodt", "Thomas Wagner", "Mina Carolina",
    "Nac Tius", "Mylius Zeramuski",

    # Marley internment zone / Liberio
    "Mr. Leonhart",

    # Special labels
    "Narrator",  # voiceover / narration
    "Crowd",     # indistinguishable crowd / background shout
    "Unknown",   # speaker truly cannot be determined
]
# fmt: on

CHAR_LIST = ", ".join(AOT_CHARACTERS)

PROMPT = """\
You are an expert on the anime Attack on Titan. Below are {n} dialogue lines from Season {season}, Episode {episode}.

Known characters: {characters}

For each numbered line, identify who is MOST LIKELY speaking. Follow these rules strictly:
- Use a name exactly as it appears in the character list above.
- "Narrator" → formal third-person voiceover describing the world, history, or events \
(e.g. "Humanity was suddenly reminded…", "Over a century ago…", "An estimated X people…"). \
PRIORITY RULE: if a block BEGINS with or is dominated by narrator-style text, label it \
"Narrator" even if a short character line appears at the end.
- "Crowd" → indistinguishable crowd noise, battle shouts, or many unnamed voices at once \
(NOT a back-and-forth conversation between named characters).
- "Unknown" → you genuinely cannot determine the speaker.
- NEVER return "Multiple". Some lines bundle several characters' dialogue together; in that \
case assign the character with the MOST spoken lines or the MOST PROMINENT speech in the block. \
If you truly cannot determine a primary speaker, use "Unknown".

Return ONLY a valid JSON object mapping each line number (as a string key) to the speaker name. \
No explanation, no markdown fences.

Example: {{"1": "Eren Yeager", "2": "Narrator", "3": "Crowd", "4": "Unknown"}}

Lines:
{lines}"""


def build_prompt(season: int, episode: int, sentences: list[str]) -> str:
    lines_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return PROMPT.format(
        n=len(sentences),
        season=season,
        episode=episode,
        characters=CHAR_LIST,
        lines=lines_text,
    )


def parse_labels(raw: str, n: int) -> list[str]:
    """Parse Gemini's JSON response into a list of n character labels."""
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:])
        raw = raw.rsplit("```", 1)[0].strip()

    parsed = json.loads(raw)

    if isinstance(parsed, dict):
        labels = [parsed.get(str(i + 1), "Unknown") for i in range(n)]
    else:
        labels = list(parsed)[:n]
        labels += ["Unknown"] * (n - len(labels))

    # Sanitise any residual "Multiple" the model may still produce
    return ["Unknown" if lbl == "Multiple" else lbl for lbl in labels]


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Set the GEMINI_API_KEY environment variable before running.")

    client = genai.Client(api_key=api_key)

    df = pd.read_csv(INPUT_FILE, dtype={"sentence": str, "episode": int, "season": int, "character": str})
    df["character"] = df["character"].fillna("")
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    # episode_meta[i] = (season, episode, original_df_indices)
    # Order must match the order requests are submitted so results line up.
    groups = list(df.groupby(["season", "episode"], sort=True))
    episode_meta = [(s, e, grp.index.tolist()) for (s, e), grp in groups]

    # ── Phase 1: submit (or resume) ───────────────────────────────────────────
    if os.path.exists(JOB_FILE):
        with open(JOB_FILE) as f:
            job_name = f.read().strip()
        print(f"Resuming existing batch job: {job_name}\n")
    else:
        requests = []
        for (season, episode), group_df in groups:
            prompt = build_prompt(season, episode, group_df["sentence"].tolist())
            requests.append({
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "config":   {"temperature": 0.0},
            })

        print(f"Submitting {len(requests)} requests (one per episode) to Batch API…")
        batch_job = client.batches.create(
            model=MODEL,
            src=requests,
            config={"display_name": "aot-transcript-labeling"},
        )
        job_name = batch_job.name
        with open(JOB_FILE, "w") as f:
            f.write(job_name)
        print(f"Job submitted: {job_name}\n")

    # ── Phase 2: poll ─────────────────────────────────────────────────────────
    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name
        print(f"[{time.strftime('%H:%M:%S')}] State: {state}")
        if state in terminal_states:
            break
        time.sleep(POLL_INTERVAL)

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"Batch job did not succeed: {batch_job.state.name}\n{batch_job.error}")

    # ── Phase 3: parse results ────────────────────────────────────────────────
    responses = batch_job.dest.inlined_responses
    if len(responses) != len(episode_meta):
        print(f"WARNING: expected {len(episode_meta)} responses, got {len(responses)}")

    for i, inline_response in enumerate(responses):
        season, episode, indices = episode_meta[i]
        n = len(indices)

        if inline_response.error:
            print(f"  ERROR S{season}E{episode:02d}: {inline_response.error} — filling Unknown")
            labels = ["Unknown"] * n
        else:
            try:
                labels = parse_labels(inline_response.response.text.strip(), n)
            except Exception as e:
                print(f"  PARSE ERROR S{season}E{episode:02d}: {e} — filling Unknown")
                labels = ["Unknown"] * n

        for idx, label in zip(indices, labels):
            df.loc[idx, "character"] = label

        print(f"  [{i + 1:03d}/{len(episode_meta)}] S{season}E{episode:02d} → {n} lines labeled")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Labeled data saved to {OUTPUT_FILE}")

    os.remove(JOB_FILE)
    print("Cleaned up job file.")


if __name__ == "__main__":
    main()
