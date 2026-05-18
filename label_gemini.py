#!/usr/bin/env python3
"""Label AoT transcript lines with character names using Gemini 2.5 Pro.

Install dependency first:
    pip install google-generativeai

Usage:
    GEMINI_API_KEY=your_key python label_gemini.py
"""

import os
import json
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

INPUT_FILE = "./data/aot.csv"
OUTPUT_FILE = "./data/aot_transcripts_labeled.csv"
MODEL = "gemini-2.5-pro"
BATCH_SIZE = 30   # lines per Gemini call
DELAY = 4.0       # seconds between API calls

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
    "Boris Feulner", "Dita Ness", "Rashad",

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
    "Tom Ksaver", "Ymir Fritz",
    "Marcel Galliard", "Bertholdt's father",

    # Paradis political / military leadership
    "Dot Pixis", "Zachary Daz", "Nile Dok",
    "Hitch Dreyse", "Marlowe Freudenberg",
    "Sandra", "Gordon", "Thomas",

    # 104th Cadet Corps peers
    "Marco Bodt", "Thomas Wagner", "Mina Carolina",
    "Nac Tius", "Mylius Zeramuski",

    # Marley internment zone / Liberio
    "Mr. Leonhart",

    # Special labels
    "Narrator",       # voiceover / narration
    "Crowd",          # indistinguishable crowd / background shout
    "Unknown",        # speaker truly cannot be determined
]
# fmt: on

PROMPT = """\
You are an expert on the anime Attack on Titan. Below are {n} dialogue lines from Season {season}, Episode {episode}.

Known characters: {characters}

For each numbered line, identify who is MOST LIKELY speaking. Follow these rules strictly:
- Use a name exactly as it appears in the character list above.
- "Narrator" → formal third-person voiceover describing the world, history, or events (e.g. "Humanity was suddenly reminded…", "Over a century ago…", "An estimated X people…"). PRIORITY RULE: if a block BEGINS with or is dominated by narrator-style text, label it "Narrator" even if a short character line appears at the end.
- "Crowd" → indistinguishable crowd noise, battle shouts, or many unnamed voices at once (NOT a back-and-forth conversation between named characters).
- "Unknown" → you genuinely cannot determine the speaker.
- NEVER return "Multiple". Some lines bundle several characters' dialogue together; in that case assign the character with the MOST spoken lines or the MOST PROMINENT speech in the block. If you truly cannot determine a primary speaker, use "Unknown".

Return ONLY a valid JSON object mapping each line number (as a string key) to the speaker name. No explanation, no markdown fences.

Example: {{"1": "Eren Yeager", "2": "Narrator", "3": "Crowd", "4": "Unknown"}}

Lines:
{lines}"""


def label_episode_batch(model, season: int, episode: int, sentences: list[str]) -> list[str]:
    characters = []
    char_list = ", ".join(AOT_CHARACTERS)

    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start : start + BATCH_SIZE]
        lines_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(batch))

        prompt = PROMPT.format(
            n=len(batch),
            season=season,
            episode=episode,
            characters=char_list,
            lines=lines_text,
        )

        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()

            # Strip markdown code fences if the model adds them
            if raw.startswith("```"):
                raw = "\n".join(raw.splitlines()[1:])
                raw = raw.rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)

            # Accept either the keyed-dict format {"1": "char", ...} or
            # a fallback positional array ["char", ...] from older model outputs.
            if isinstance(parsed, dict):
                batch_labels = [parsed.get(str(i + 1), "Unknown") for i in range(len(batch))]
            else:
                batch_labels = list(parsed)
                if len(batch_labels) != len(batch):
                    print(f"    WARNING: expected {len(batch)} labels, got {len(batch_labels)} — padding with Unknown")
                    batch_labels += ["Unknown"] * (len(batch) - len(batch_labels))
                batch_labels = batch_labels[: len(batch)]

            # Replace any residual "Multiple" labels Gemini may still produce.
            batch_labels = ["Unknown" if lbl == "Multiple" else lbl for lbl in batch_labels]

            characters.extend(batch_labels)

        except Exception as e:
            print(f"    ERROR on batch starting at line {start + 1}: {e}")
            characters.extend(["Unknown"] * len(batch))

        if start + BATCH_SIZE < len(sentences):
            time.sleep(DELAY)

    return characters


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Set the GEMINI_API_KEY environment variable before running.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    df = pd.read_csv(INPUT_FILE, dtype={"sentence": str, "episode": int, "season": int, "character": str})
    df["character"] = df["character"].fillna("")
    print(f"Loaded {len(df)} rows from {INPUT_FILE}\n")

    groups = list(df.groupby(["season", "episode"], sort=True))
    total = len(groups)

    for i, ((season, episode), group_df) in enumerate(groups, 1):
        print(f"[{i:03d}/{total}] Season {season} Episode {episode:02d}  ({len(group_df)} lines)")
        sentences = group_df["sentence"].tolist()
        labels = label_episode_batch(model, season, episode, sentences)
        df.loc[group_df.index, "character"] = labels

        # Save after each episode so partial work is not lost
        df.to_csv(OUTPUT_FILE, index=False)

        if i < total:
            time.sleep(DELAY)

    print(f"\nDone. Labeled data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
