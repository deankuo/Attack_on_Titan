"""
Comments-only scraper for r/attackontitan, aligned to episode airdate windows.

Companion to episode_scraper.py: reuses the same episode metadata and 7-day
window definition, but ONLY fetches comments (skips submissions entirely).
Outputs are schema-compatible with episode_scraper.py's comments CSV, so you
can drop them into the same analysis pipeline.

Why a separate script?
  episode_scraper.py's scrape_log tracks completed (season, episode) pairs
  but doesn't distinguish posts vs comments. After running posts-only across
  all 89 episodes, the log says "everything done" and --resume would skip
  everything. This script keeps its own log (scrape_log_comments.csv) so it
  can run independently and resume safely.

Usage:
    # Full run — comments for every episode window (4-8 hours)
    python comments_scraper.py

    # Resume after interruption (safe to re-run)
    python comments_scraper.py --resume

    # Pilot on one episode
    python comments_scraper.py --season 1 --episode 25

    # Just one season
    python comments_scraper.py --season 4 --resume

    # Build comment trees per episode (one JSON per episode)
    python comments_scraper.py --trees

Outputs (in data/episode_scrape/):
    comments_by_episode.csv    one row per comment, with episode context cols
    scrape_log_comments.csv    which episodes have completed comment scrape
    trees/S{ss}E{ee}.json      (only if --trees) nested comment trees
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from pullpush_scraper import (
    iter_items,
    comment_dict_to_row,
    DEFAULT_UA,
    CommentRow,
)
from episode_scraper import (
    load_episodes,
    compute_window,
    build_comment_tree,
    WideCsvWriter,
    METADATA_PATH,
    DEFAULT_WINDOW_DAYS,
    SUBREDDIT,
    EPISODE_CONTEXT_COLS,
    ANALYSIS_STUB_COLS,
)


# ---------------------------------------------------------------------------
# Independent scrape log (different filename so it doesn't collide with the
# posts log)
# ---------------------------------------------------------------------------


def load_comments_log(path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row["season"]), int(row["episode"])))
            except (KeyError, ValueError):
                continue
    return done


def append_comments_log(path: Path, row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Per-episode comment scrape
# ---------------------------------------------------------------------------


def scrape_episode_comments(
    ep: dict,
    next_ep: Optional[dict],
    window_days: int,
    writer: WideCsvWriter,
    trees_dir: Optional[Path],
    user_agent: str,
    sleep_between: float,
    ignore_next_airdate: bool = False,
) -> dict:
    start_dt, end_dt = compute_window(
        ep, next_ep, window_days, ignore_next_airdate=ignore_next_airdate
    )
    after = int(start_dt.timestamp())
    before = int(end_dt.timestamp())
    s, e = ep["season"], ep["episode"]
    cum = ep["cumulative"]

    ctx_common = {
        "ep_season": s,
        "ep_episode": e,
        "ep_cumulative": cum,
        "ep_title": ep["title"],
        "ep_airdate": ep["airdate"],
        "window_start_iso": start_dt.isoformat(),
        "window_end_iso": end_dt.isoformat(),
    }

    n_comments = 0
    total_score = 0
    authors: set[str] = set()
    raw_records: list[dict] = []  # only kept if we're building trees

    if start_dt >= end_dt:
        # Same-day next-episode edge case: window collapsed. Skip cleanly.
        return {
            "season": s, "episode": e, "cumulative": cum,
            "title": ep["title"], "airdate": ep["airdate"],
            "window_start": start_dt.isoformat(),
            "window_end": end_dt.isoformat(),
            "n_comments": 0, "total_comment_score": 0,
            "n_unique_comment_authors": 0,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "note": "window collapsed (next episode aired same day)",
        }

    for raw in iter_items(
        kind="comment",
        subreddit=SUBREDDIT,
        after=after,
        before=before,
        query=None,
        user_agent=user_agent,
        sleep_between=sleep_between,
    ):
        crow = comment_dict_to_row(raw)
        d = asdict(crow)
        days_since = (d["created_utc"] - start_dt.timestamp()) / 86400.0
        ep_ctx = {
            **ctx_common,
            "days_since_airdate": round(days_since, 4),
            "is_megathread_candidate": False,  # only meaningful on posts
        }
        writer.write(d, ep_ctx)
        n_comments += 1
        total_score += d["score"]
        authors.add(d["author"])
        if trees_dir is not None:
            raw_records.append(raw)

    # Build per-episode comment tree JSON
    if trees_dir is not None and raw_records:
        trees_dir.mkdir(parents=True, exist_ok=True)
        by_post: dict[str, list[dict]] = defaultdict(list)
        for raw in raw_records:
            lid = raw.get("link_id") or ""
            pid = lid[3:] if lid.startswith("t3_") else lid
            if pid:
                by_post[pid].append(raw)
        tree_doc = {
            "episode": {
                "season": s, "episode": e, "cumulative": cum,
                "title": ep["title"], "airdate": ep["airdate"],
                "window_start": start_dt.isoformat(),
                "window_end": end_dt.isoformat(),
            },
            "comments_by_post": {
                pid: build_comment_tree(cmts)
                for pid, cmts in by_post.items()
            },
        }
        tree_path = trees_dir / f"S{s:02d}E{e:02d}.json"
        with tree_path.open("w", encoding="utf-8") as f:
            json.dump(tree_doc, f, ensure_ascii=False, indent=2)

    return {
        "season": s,
        "episode": e,
        "cumulative": cum,
        "title": ep["title"],
        "airdate": ep["airdate"],
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        "n_comments": n_comments,
        "total_comment_score": total_score,
        "n_unique_comment_authors": len(authors - {"[deleted]"}),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "note": "",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comments-only scraper aligned to episode airdate windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metadata", default=str(METADATA_PATH))
    p.add_argument("--output-dir", default="data/episode_scrape")
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    p.add_argument("--ignore-next-airdate", action="store_true",
                   help="Don't cap the window at the next episode's airdate. "
                        "Useful for episodes whose next-episode aired the "
                        "same day (window otherwise collapses to 0).")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--trees", action="store_true",
                   help="Build per-episode JSON comment trees")
    p.add_argument("--resume", action="store_true",
                   help="Skip episodes already in scrape_log_comments.csv")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="Seconds between PullPush page requests (gentler default "
                        "than the posts run because comment volume is much higher)")
    p.add_argument("--user-agent", default=DEFAULT_UA)
    return p.parse_args()


def select_episodes(episodes: list[dict], args) -> list[int]:
    idxs = list(range(len(episodes)))
    if args.season is not None:
        idxs = [i for i in idxs if episodes[i]["season"] == args.season]
    if args.episode is not None:
        idxs = [i for i in idxs if episodes[i]["episode"] == args.episode]
    return idxs


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(Path(args.metadata))
    selected = select_episodes(episodes, args)
    print(f"[comments-scraper] loaded {len(episodes)} episodes, "
          f"{len(selected)} selected, sleep={args.sleep}s")

    if args.dry_run:
        print(f"{'idx':>4} {'S/E':<6} {'airdate':<10} {'window_start':<12} "
              f"{'window_end':<12}  title")
        for i in selected:
            ep = episodes[i]
            nxt = episodes[i + 1] if i + 1 < len(episodes) else None
            s_dt, e_dt = compute_window(
                ep, nxt, args.window_days,
                ignore_next_airdate=args.ignore_next_airdate,
            )
            print(f"{i:>4} S{ep['season']:02d}E{ep['episode']:02d} "
                  f"{ep['airdate']:<10} {s_dt.date().isoformat():<12} "
                  f"{e_dt.date().isoformat():<12}  {ep['title'][:60]}")
        return 0

    comments_path = out / "comments_by_episode.csv"
    summary_path = out / "per_episode_comments_summary.csv"
    log_path = out / "scrape_log_comments.csv"
    trees_dir = (out / "trees") if args.trees else None

    comment_fields = [f.name for f in CommentRow.__dataclass_fields__.values()]
    writer = WideCsvWriter(comments_path, comment_fields)

    done = load_comments_log(log_path) if args.resume else set()
    if done:
        print(f"[comments-scraper] resuming, skipping {len(done)} already-done episodes")

    summary_rows: list[dict] = []
    try:
        for i in tqdm(selected, desc="episodes", unit="ep"):
            ep = episodes[i]
            if (ep["season"], ep["episode"]) in done:
                continue
            nxt = episodes[i + 1] if i + 1 < len(episodes) else None
            try:
                summary = scrape_episode_comments(
                    ep=ep,
                    next_ep=nxt,
                    window_days=args.window_days,
                    writer=writer,
                    trees_dir=trees_dir,
                    user_agent=args.user_agent,
                    sleep_between=args.sleep,
                    ignore_next_airdate=args.ignore_next_airdate,
                )
                summary_rows.append(summary)
                append_comments_log(log_path, {
                    "season": ep["season"],
                    "episode": ep["episode"],
                    "cumulative": ep["cumulative"],
                    "n_comments": summary["n_comments"],
                    "scraped_at": summary["scraped_at"],
                    "note": summary.get("note", ""),
                })
            except Exception as exc:
                print(f"[comments-scraper] ERROR on S{ep['season']:02d}E{ep['episode']:02d}: {exc}",
                      file=sys.stderr)
                continue
    finally:
        writer.close()

    # Merge summary rows with any existing per_episode_comments_summary
    if summary_rows:
        existing: dict[tuple[int, int], dict] = {}
        if summary_path.exists():
            with summary_path.open() as f:
                for row in csv.DictReader(f):
                    existing[(int(row["season"]), int(row["episode"]))] = row
        for r in summary_rows:
            existing[(r["season"], r["episode"])] = {k: str(v) for k, v in r.items()}
        merged = sorted(existing.values(),
                        key=lambda r: (int(r["season"]), int(r["episode"])))
        fields = list(summary_rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in merged:
                w.writerow({k: r.get(k, "") for k in fields})

    print("\n[comments-scraper] done.")
    print(f"  comments: {comments_path}")
    print(f"  log:      {log_path}")
    print(f"  summary:  {summary_path}")
    if trees_dir is not None:
        print(f"  trees:    {trees_dir}/S{{ss}}E{{ee}}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
