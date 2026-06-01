# How to run the scrapers yourself

Everything you need to drive the scrapers from your own Terminal. Same machine, same data; you're just taking the wheel.

## One-time setup (already done, but here for reference)

```bash
cd ~/Desktop/SICSS/aot_web_scapper
python3 -m venv .venv          # already exists
source .venv/bin/activate      # activate it
pip install -r requirements.txt # already installed
```

The `.venv` is already created and packages are installed. Every new Terminal session needs the `source .venv/bin/activate` line so `python` points to the venv.

## What's already been scraped

| file | rows | what it is |
|------|------|------------|
| `data/episode_scrape/posts_by_episode.csv` | 52,317 | every post from each episode's 7-day window |
| `data/episode_scrape/per_episode_summary.csv` | 89 | one row per episode with aggregate stats |
| `data/episode_scrape/scrape_log.csv` | 89 | which episodes finished the posts run |

Three episodes have **0 posts**: S01E01, S01E06 (early 2013, sub was tiny), and **S04E14** (window collapsed to 0 because S04E15 aired the same day — that's a bug we now have a flag for).

## Commands you'll want

Every command assumes you've activated the venv first (`source .venv/bin/activate`) and you're in `~/Desktop/SICSS/aot_web_scapper`.

### 1. Backfill the three zero-post episodes with a doubled window

This uses the new `--ignore-next-airdate` flag (so S04E14's window doesn't collapse) plus `--window-days 14` (double the default 7).

```bash
# S04E14 — fixes the same-day collapse bug
python episode_scraper.py --no-comments --window-days 14 --ignore-next-airdate \
    --season 4 --episode 14

# S01E01 and S01E06 — wider net for sparse 2013 data
python episode_scraper.py --no-comments --window-days 14 --ignore-next-airdate \
    --season 1 --episode 1
python episode_scraper.py --no-comments --window-days 14 --ignore-next-airdate \
    --season 1 --episode 6
```

Each takes well under a minute. New posts are appended to `posts_by_episode.csv`; the `ep_season`/`ep_episode` columns make filtering trivial later in pandas.

**Note on overlap:** doubled windows will overlap with the next episode's window. Posts in the overlap appear twice — once tagged for each episode. If you want exclusive episode-tagging, dedupe by `post_id` keeping the first occurrence (which is the episode the post actually belongs to chronologically).

### 2. Run the full comments scrape (all 89 episodes)

```bash
python comments_scraper.py --resume --sleep 1.0
```

Budget **4–8 hours**. `--resume` is safe — it skips episodes already in `scrape_log_comments.csv`, so you can Ctrl+C and restart any time. `--sleep 1.0` is gentle on PullPush; bump to 1.5 if you hit lots of 429s, drop to 0.5 if PullPush is being friendly.

To prevent your Mac from sleeping during the long run, prefix with `caffeinate`:

```bash
caffeinate -i python comments_scraper.py --resume --sleep 1.0
```

(Ctrl+C still works; `caffeinate` just keeps the display/system awake.)

### 3. Run comments for a single season

```bash
python comments_scraper.py --season 4 --resume
```

Useful if you want to start with the highest-engagement season and decide later whether the older ones are worth the time.

### 4. Run comments for one specific episode

```bash
python comments_scraper.py --season 4 --episode 30
```

Useful for spot-checking or filling a gap.

### 5. Optional: build per-episode JSON comment trees

```bash
python comments_scraper.py --resume --trees
```

Adds `--trees` so each episode also produces `data/episode_scrape/trees/S{ss}E{ee}.json` with the full reply tree (good for conversation/network analysis). Trees aren't built by default because they're slow on heavy episodes.

### 6. Other useful flags (every scraper accepts these)

| flag | meaning |
|------|---------|
| `--dry-run` | print which episodes/windows would be scraped, don't fetch |
| `--season N` | only this season |
| `--episode N` | only this episode (combine with `--season`) |
| `--resume` | skip episodes already in the relevant `scrape_log*.csv` |
| `--sleep S` | seconds between PullPush page requests |
| `--window-days N` | window length (default 7) |
| `--ignore-next-airdate` | don't cap the window at the next episode's airdate |

## What to do if things go wrong

**Rate limits (429s).** The script auto-retries with exponential backoff (2, 4, 8, 16, 32s) and gives up after 5 tries. If an episode fails, you'll see `ERROR on S0XE0Y: ...` in the output. Re-run with `--resume` and a higher `--sleep` to retry the failed ones.

**"command not found: python".** You forgot `source .venv/bin/activate`. Run it.

**PullPush is down.** It's community-run. Check https://pullpush.io or just wait an hour and retry — `--resume` will pick up where it left off.

**Duplicate rows in your CSV.** Happens when an episode partly succeeds, fails on a later page, and gets re-fetched. Dedupe in pandas:

```python
import pandas as pd
df = pd.read_csv("data/episode_scrape/posts_by_episode.csv")
df = df.drop_duplicates(subset=["post_id"], keep="last")
```

Or the equivalent for comments by `comment_id`.

## Sanity-check the outputs

```bash
python -c "
import csv
with open('data/episode_scrape/per_episode_summary.csv') as f:
    rows = list(csv.DictReader(f))
print(f'episodes: {len(rows)}')
print(f'total posts: {sum(int(r[\"n_posts\"]) for r in rows):,}')
print(f'zero-post episodes: {[(int(r[\"season\"]), int(r[\"episode\"])) for r in rows if int(r[\"n_posts\"]) == 0]}')
"
```
