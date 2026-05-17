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

INPUT_FILE = "aot_transcripts_raw.csv"
OUTPUT_FILE = "aot_transcripts_labeled.csv"
MODEL = "gemini-2.5-pro"
BATCH_SIZE = 30   # lines per Gemini call
DELAY = 4.0       # seconds between API calls

# fmt: off
AOT_CHARACTERS = [
    # Core Survey Corps
    "Eren Yeager", "Mikasa Ackerman", "Armin Arlert", "Levi Ackerman",
    "Hange Zoë", "Erwin Smith", "Jean Kirstein", "Connie Springer",
    "Sasha Blouse", "Historia Reiss", "Ymir", "Moblit Berner",
    "Squad Leader Mike", "Floch Forster", "Isabel Magnolia", "Furlan Church",
    # Garrison / Military Police
    "Hannes", "Dot Pixis", "Keith Shadis", "Nile Dok", "Kenny Ackerman",
    # Warriors / Marleyans
    "Reiner Braun", "Bertholdt Hoover", "Annie Leonhart",
    "Zeke Yeager", "Pieck Finger", "Porco Galliard", "Gabi Braun", "Falco Grice",
    # Yeager family / supporting
    "Grisha Yeager", "Carla Yeager", "Dina Fritz",
    # Marley allies / others
    "Yelena", "Onyankopon", "Rod Reiss", "Pastor Nick",
    # Special labels
    "Narrator",     # voiceover / narration
    "Multiple",     # line contains clearly mixed speakers
    "crowd line",   # indistinguishable crowd / background shout
    "Unknown",      # speaker truly cannot be determined
]
# fmt: on

PROMPT = """\
You are an expert on the anime Attack on Titan. Below are {n} dialogue lines from Season {season}, Episode {episode}.

Known characters: {characters}

For each numbered line, identify who is most likely speaking. Follow these rules strictly:
- Use a name exactly as written in the character list above.
- "Narrator" → voiceover/narration with no identifiable speaker.
- "Multiple" → the line contains more than one distinct speaker mixed together.
- "crowd line" → indistinguishable crowd shout, battle noise, or background voices.
- "Unknown" → you genuinely cannot identify the speaker.

Return ONLY a valid JSON array of {n} strings, one per line, in the same order. No explanation, no markdown fences.

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

            batch_labels = json.loads(raw)

            if len(batch_labels) != len(batch):
                print(f"    WARNING: expected {len(batch)} labels, got {len(batch_labels)} — padding with Unknown")
                batch_labels += ["Unknown"] * (len(batch) - len(batch_labels))

            characters.extend(batch_labels[: len(batch)])

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
