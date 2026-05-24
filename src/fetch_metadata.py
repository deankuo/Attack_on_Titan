#!/usr/bin/env python3
"""Fetch Attack on Titan episode metadata from the TVmaze API.

No API key required. Single request fetches all episodes at once.
Output: aot_episode_metadata.csv

Columns: season, episode, title, airdate, runtime, rating, summary, tvmaze_id
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup

SHOW_ID = 919  # Attack on Titan (2013) on TVmaze
OUTPUT_FILE = "./data/aot_episode_metadata.csv"


def strip_html(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ").strip()


def main():
    print(f"Fetching metadata from https://api.tvmaze.com/shows/{SHOW_ID}/episodes ...")
    resp = requests.get(f"https://api.tvmaze.com/shows/{SHOW_ID}/episodes", timeout=15)
    resp.raise_for_status()
    episodes = resp.json()
    print(f"Received {len(episodes)} episodes.\n")

    rows = []
    for ep in episodes:
        rows.append({
            "season":    ep["season"],
            "episode":   ep["number"],
            "title":     ep["name"],
            "airdate":   ep.get("airdate", ""),
            "runtime":   ep.get("runtime", ""),
            "rating":    ep.get("rating", {}).get("average", ""),
            "summary":   strip_html(ep.get("summary")),
            "tvmaze_id": ep["id"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    print(f"Seasons:  {sorted(df['season'].unique())}")
    print(f"Episodes per season: { dict(df.groupby('season')['episode'].count()) }")


if __name__ == "__main__":
    main()
