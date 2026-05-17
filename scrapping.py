#!/usr/bin/env python3
"""Scrape Attack on Titan transcripts from springfieldspringfield.co.uk."""

import re
import time
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup, NavigableString

BASE_URL = "https://www.springfieldspringfield.co.uk"
LISTING_URL = f"{BASE_URL}/episode_scripts.php?tv-show=attack-on-titan-2013"
EPISODE_URL = f"{BASE_URL}/view_episode_scripts.php?tv-show=attack-on-titan-2013&episode="
OUTPUT_FILE = "./data/aot_transcripts_raw.csv"
TRANSCRIPT_DIR = Path("transcript")
DELAY = 2.5  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    )
}


def get_episode_list():
    resp = requests.get(LISTING_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    episodes = []
    for link in soup.select("a[href*=view_episode_scripts]"):
        href = link["href"]
        m = re.search(r"episode=(s(\d+)e(\d+))", href)
        if m:
            code = m.group(1)
            season = int(m.group(2))
            episode = int(m.group(3))
            title = link.get_text(strip=True)
            episodes.append((code, season, episode, title))

    return episodes


def get_transcript_lines(episode_code):
    resp = requests.get(EPISODE_URL + episode_code, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    container = soup.find("div", class_="scrolling-script-container")
    if not container:
        return []

    lines = []
    for child in container.children:
        if not isinstance(child, NavigableString):
            continue
        text = child.strip()
        if text:
            lines.append(text)

    return lines


def save_txt(code: str, lines: list[str]) -> None:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    (TRANSCRIPT_DIR / f"{code}.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Fetching episode list...")
    episodes = get_episode_list()
    print(f"Found {len(episodes)} episodes.\n")

    rows = []
    for i, (code, season, episode, title) in enumerate(episodes, 1):
        print(f"[{i:02d}/{len(episodes)}] S{season:02d}E{episode:02d} — {title}")
        try:
            lines = get_transcript_lines(code)
            print(f"         {len(lines)} lines scraped")
            save_txt(code, lines)
            for line in lines:
                rows.append({
                    "sentence": line,
                    "episode": episode,
                    "season": season,
                    "character": "",
                })
        except Exception as e:
            print(f"         ERROR: {e}")

        # Save progress after every episode so a crash doesn't lose work
        if rows:
            pd.DataFrame(rows, columns=["sentence", "episode", "season", "character"]).to_csv(
                OUTPUT_FILE, index=False
            )

        if i < len(episodes):
            time.sleep(DELAY)

    df = pd.DataFrame(rows, columns=["sentence", "episode", "season", "character"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(df)} total rows saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
