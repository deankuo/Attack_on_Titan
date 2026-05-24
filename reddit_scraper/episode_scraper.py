"""
Episode-windowed Reddit scraper for r/attackontitan.

For each episode in aot_episode_metadata.csv, scrape every post and comment
made during the discussion window:

    window = [airdate, min(airdate + window_days, next_airdate))

Median gap between AoT episodes is 7 days, so windows naturally cap at one
week even when the next episode is months away (hiatus periods). The final
episode gets a flat 7-day window.

Outputs (all in --output-dir, default `data/episode_scrape/`):

    posts_by_episode.csv          one row per post, with episode metadata
    comments_by_episode.csv       one row per comment, with episode metadata
    per_episode_summary.csv       aggregate stats per episode
    trees/S{ss}E{ee}.json         per-episode JSON with nested comment trees
    scrape_log.csv                which episodes have been scraped (for resume)

Posts/comments CSVs also include empty sentiment_score / sentiment_label /
topic_label / notes columns so you can fill them downstream without altering
the schema.

Usage:
    # Full run — all 89 episodes, posts + comments + trees + summary
    python episode_scraper.py

    # Just show the windows that would be scraped, don't fetch
    python episode_scraper.py --dry-run

    # Only one season
    python episode_scraper.py --season 4

    # Skip episodes we've already scraped (resumes from scrape_log.csv)
    python episode_scraper.py --resume

    # Posts only, faster
    python episode_scraper.py --no-comments

    # Override window length
    python episode_scraper.py --window-days 14
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm

from pullpush_scraper import (
    iter_items,
    post_dict_to_row,
    comment_dict_to_row,
    DEFAULT_UA,
)


METADATA_PATH = Path("data/aot_episode_metadata.csv")
DEFAULT_WINDOW_DAYS = 7
SUBREDDIT = "attackontitan"


# Extra columns appended to every post / comment row so episode context is
# preserved on the row itself (makes per-episode filtering trivial in pandas)
EPISODE_CONTEXT_COLS = [
    "ep_season", "ep_episode", "ep_cumulative", "ep_title",
    "ep_airdate", "window_start_iso", "window_end_iso",
    "days_since_airdate", "is_megathread_candidate",
]
# Stub columns for downstream NLP — left empty by the scraper
ANALYSIS_STUB_COLS = ["sentiment_score", "sentiment_label", "topic_label", "notes"]


# ---------------------------------------------------------------------------
# Episode metadata loading & window computation
# ---------------------------------------------------------------------------


def load_episodes(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["season"] = int(r["season"])
        r["episode"] = int(r["episode"])
        r["_airdate_dt"] = datetime.strptime(
            r["airdate"], "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc)
    rows.sort(key=lambda r: r["_airdate_dt"])
    # Cumulative episode index (1-based across all seasons)
    for i, r in enumerate(rows):
        r["cumulative"] = i + 1
    return rows


def compute_window(
    ep: dict,
    next_ep: Optional[dict],
    window_days: int,
    ignore_next_airdate: bool = False,
) -> tuple[datetime, datetime]:
    """Window for the episode discussion period.

    Default: [airdate, min(airdate + window_days, next_episode_airdate)).
    If ignore_next_airdate=True, we always use the full window_days. Useful
    for episodes whose next-episode aired the same day (which otherwise
    collapses the window to zero duration), and for getting a wider net
    around episodes with very low post counts.
    """
    start = ep["_airdate_dt"]
    natural_end = start + timedelta(days=window_days)
    if next_ep is not None and not ignore_next_airdate:
        natural_end = min(natural_end, next_ep["_airdate_dt"])
    return start, natural_end


# ---------------------------------------------------------------------------
# Megathread heuristic
# ---------------------------------------------------------------------------


_MEGATHREAD_KEYWORDS = ("discussion", "thread", "megathread")


def is_megathread_candidate(
    title: str,
    is_stickied: bool,
    season: int,
    episode: int,
    cumulative: int,
) -> bool:
    """Heuristic flag for the official episode discussion thread.

    A title qualifies if it (a) references the right episode by season+episode
    or cumulative number, AND (b) reads like a discussion thread. Stickied
    posts that also mention 'episode' get a more lenient pass since mods often
    sticky the megathread with just the title.
    """
    if not title:
        return False
    t = title.lower()
    refs_episode = (
        re.search(rf"\bs0?{season}\s*[-\s]*e0?{episode}\b", t)
        or re.search(rf"season\s+{season}\D+episode\s+{episode}\b", t)
        or re.search(rf"\bepisode\s+0?{episode}\b", t)
        or re.search(rf"\bep\.?\s*0?{episode}\b", t)
        or re.search(rf"\bepisode\s+{cumulative}\b", t)
    )
    if not refs_episode:
        return False
    looks_like_thread = any(k in t for k in _MEGATHREAD_KEYWORDS)
    if looks_like_thread:
        return True
    # Lenient: stickied + says "episode N" = probably the megathread even
    # without the word "discussion"
    if is_stickied:
        return True
    return False


# ---------------------------------------------------------------------------
# CSV writer with extra columns
# ---------------------------------------------------------------------------


class WideCsvWriter:
    """CSV writer that knows the full column order up front (base dataclass
    fields + episode context + analysis stubs) so all rows share a schema."""

    def __init__(self, path: Path, base_fields: list[str]):
        self.path = path
        self.fieldnames = base_fields + EPISODE_CONTEXT_COLS + ANALYSIS_STUB_COLS
        self._fh = path.open("a", newline="", encoding="utf-8")
        # Write header only if file is empty (so --resume can append)
        if path.stat().st_size == 0:
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
            self._writer.writeheader()
        else:
            self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)

    def write(self, base_row_dict: dict, ep_ctx: dict) -> None:
        merged = {**base_row_dict, **ep_ctx}
        # Ensure stub cols exist
        for k in ANALYSIS_STUB_COLS:
            merged.setdefault(k, "")
        # Drop any unexpected keys to avoid DictWriter errors
        row = {k: merged.get(k, "") for k in self.fieldnames}
        self._writer.writerow(row)

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Per-episode JSON tree builder
# ---------------------------------------------------------------------------


def build_comment_tree(comments: list[dict]) -> list[dict]:
    """Given a flat list of comment dicts (each with id, parent_id, body, ...),
    return a list of root nodes with nested 'replies' lists."""
    by_id: dict[str, dict] = {}
    for c in comments:
        c2 = dict(c)
        c2["replies"] = []
        by_id[c2["id"]] = c2

    roots: list[dict] = []
    for c in by_id.values():
        parent = c.get("parent_id", "") or ""
        if parent.startswith("t1_"):
            pid = parent[3:]
            if pid in by_id:
                by_id[pid]["replies"].append(c)
                continue
        # Top-level (parent is the post) or orphaned (parent missing) → root
        roots.append(c)
    # Sort each level by created_utc for deterministic output
    def _sort(nodes: list[dict]):
        nodes.sort(key=lambda n: n.get("created_utc") or 0)
        for n in nodes:
            _sort(n["replies"])
    _sort(roots)
    return roots


# ---------------------------------------------------------------------------
# Scrape log (for --resume)
# ---------------------------------------------------------------------------


def load_scrape_log(path: Path) -> set[tuple[int, int]]:
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


def append_scrape_log(path: Path, row: dict) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Per-episode scrape
# ---------------------------------------------------------------------------


def scrape_episode(
    ep: dict,
    next_ep: Optional[dict],
    window_days: int,
    posts_writer: WideCsvWriter,
    comments_writer: Optional[WideCsvWriter],
    trees_dir: Path,
    user_agent: str,
    sleep_between: float,
    write_tree: bool,
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

    # --- posts ---
    post_records: list[dict] = []
    post_authors: set[str] = set()
    megathread_count = 0
    top_post = None  # (score, id, title)

    for raw in iter_items(
        kind="submission",
        subreddit=SUBREDDIT,
        after=after,
        before=before,
        query=None,
        user_agent=user_agent,
        sleep_between=sleep_between,
    ):
        post_row = post_dict_to_row(raw)
        d = asdict(post_row)

        is_mt = is_megathread_candidate(
            d.get("title", ""), bool(raw.get("stickied")), s, e, cum
        )
        if is_mt:
            megathread_count += 1

        days_since = (d["created_utc"] - start_dt.timestamp()) / 86400.0
        ep_ctx = {
            **ctx_common,
            "days_since_airdate": round(days_since, 4),
            "is_megathread_candidate": is_mt,
        }
        posts_writer.write(d, ep_ctx)

        post_authors.add(d["author"])
        if top_post is None or d["score"] > top_post[0]:
            top_post = (d["score"], d["post_id"], d["title"])

        post_records.append({"raw": raw, "row": d, "is_megathread": is_mt})

    # --- comments ---
    comment_records: list[dict] = []
    total_comment_score = 0
    comment_authors: set[str] = set()

    if comments_writer is not None:
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
            comments_writer.write(d, ep_ctx)
            total_comment_score += d["score"]
            comment_authors.add(d["author"])
            comment_records.append(raw)

    # --- per-episode JSON tree ---
    if write_tree:
        trees_dir.mkdir(parents=True, exist_ok=True)
        # Group comments by post
        by_post: dict[str, list[dict]] = defaultdict(list)
        for raw in comment_records:
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
            "posts": [
                {
                    "post": pr["row"],
                    "is_megathread_candidate": pr["is_megathread"],
                    "comments": build_comment_tree(by_post.get(pr["row"]["post_id"], [])),
                }
                for pr in post_records
            ],
        }
        tree_path = trees_dir / f"S{s:02d}E{e:02d}.json"
        with tree_path.open("w", encoding="utf-8") as f:
            json.dump(tree_doc, f, ensure_ascii=False, indent=2)

    # --- summary row ---
    n_posts = len(post_records)
    avg_post_score = (
        sum(pr["row"]["score"] for pr in post_records) / n_posts if n_posts else 0
    )
    summary = {
        "season": s,
        "episode": e,
        "cumulative": cum,
        "title": ep["title"],
        "airdate": ep["airdate"],
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        "window_days": round((end_dt - start_dt).total_seconds() / 86400, 3),
        "n_posts": n_posts,
        "n_comments": len(comment_records),
        "n_megathread_candidates": megathread_count,
        "n_unique_post_authors": len(post_authors - {"[deleted]"}),
        "n_unique_comment_authors": len(comment_authors - {"[deleted]"}),
        "avg_post_score": round(avg_post_score, 2),
        "total_comment_score": total_comment_score,
        "top_post_id": top_post[1] if top_post else "",
        "top_post_title": top_post[2] if top_post else "",
        "top_post_score": top_post[0] if top_post else 0,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Episode-windowed scraper for r/attackontitan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metadata", default=str(METADATA_PATH),
                   help="Path to aot_episode_metadata.csv")
    p.add_argument("--output-dir", default="data/episode_scrape",
                   help="Where to write outputs")
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                   help="Cap window at this many days past airdate (also used "
                        "for the final episode with no next airdate)")
    p.add_argument("--season", type=int, default=None,
                   help="Only scrape this season")
    p.add_argument("--episode", type=int, default=None,
                   help="Only scrape this episode number (combine with --season)")
    p.add_argument("--no-comments", action="store_true",
                   help="Skip comments (much faster, no trees)")
    p.add_argument("--no-trees", action="store_true",
                   help="Skip per-episode JSON tree files")
    p.add_argument("--ignore-next-airdate", action="store_true",
                   help="Don't cap the window at the next episode's airdate. "
                        "Use the full --window-days regardless. Useful for "
                        "episodes whose next-episode aired the same day "
                        "(otherwise the window collapses to 0).")
    p.add_argument("--resume", action="store_true",
                   help="Skip episodes already in scrape_log.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the windows that would be scraped and exit")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds between PullPush page requests")
    p.add_argument("--user-agent", default=DEFAULT_UA)
    return p.parse_args()


def select_episodes(episodes: list[dict], args) -> list[int]:
    """Return indices into `episodes` that we should scrape."""
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
    print(f"[episode-scraper] loaded {len(episodes)} episodes, {len(selected)} selected")

    if args.dry_run:
        print(f"{'idx':>4} {'S/E':<6} {'cum':>3} {'airdate':<10} "
              f"{'window_start':<20} {'window_end':<20} {'days':>5}  title")
        for i in selected:
            ep = episodes[i]
            nxt = episodes[i + 1] if i + 1 < len(episodes) else None
            s_dt, e_dt = compute_window(
                ep, nxt, args.window_days,
                ignore_next_airdate=args.ignore_next_airdate,
            )
            print(f"{i:>4} S{ep['season']:02d}E{ep['episode']:02d}  "
                  f"{ep['cumulative']:>3} {ep['airdate']:<10} "
                  f"{s_dt.date().isoformat():<20} {e_dt.date().isoformat():<20} "
                  f"{(e_dt - s_dt).days:>5}  {ep['title'][:60]}")
        return 0

    posts_path = out / "posts_by_episode.csv"
    comments_path = out / "comments_by_episode.csv"
    summary_path = out / "per_episode_summary.csv"
    trees_dir = out / "trees"
    scrape_log_path = out / "scrape_log.csv"

    # Base fieldnames pulled from the dataclasses in pullpush_scraper
    from pullpush_scraper import PostRow, CommentRow
    post_fields = [f.name for f in PostRow.__dataclass_fields__.values()]
    comment_fields = [f.name for f in CommentRow.__dataclass_fields__.values()]

    posts_writer = WideCsvWriter(posts_path, post_fields)
    comments_writer = (
        None if args.no_comments else WideCsvWriter(comments_path, comment_fields)
    )

    done = load_scrape_log(scrape_log_path) if args.resume else set()
    if done:
        print(f"[episode-scraper] resuming, skipping {len(done)} already-scraped episodes")

    summary_rows: list[dict] = []
    try:
        for i in tqdm(selected, desc="episodes", unit="ep"):
            ep = episodes[i]
            if (ep["season"], ep["episode"]) in done:
                continue
            nxt = episodes[i + 1] if i + 1 < len(episodes) else None
            try:
                summary = scrape_episode(
                    ep=ep,
                    next_ep=nxt,
                    window_days=args.window_days,
                    posts_writer=posts_writer,
                    comments_writer=comments_writer,
                    trees_dir=trees_dir,
                    user_agent=args.user_agent,
                    sleep_between=args.sleep,
                    write_tree=(not args.no_trees and not args.no_comments),
                    ignore_next_airdate=args.ignore_next_airdate,
                )
                summary_rows.append(summary)
                append_scrape_log(scrape_log_path, {
                    "season": ep["season"],
                    "episode": ep["episode"],
                    "cumulative": ep["cumulative"],
                    "n_posts": summary["n_posts"],
                    "n_comments": summary["n_comments"],
                    "scraped_at": summary["scraped_at"],
                })
            except Exception as exc:
                print(f"[episode-scraper] ERROR on S{ep['season']:02d}E{ep['episode']:02d}: {exc}",
                      file=sys.stderr)
                continue
    finally:
        posts_writer.close()
        if comments_writer is not None:
            comments_writer.close()

    # Write summary (full rewrite — always reflects this run)
    if summary_rows:
        # Append-or-create behavior: load existing, merge by (season, episode)
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

    print("\n[episode-scraper] done.")
    print(f"  posts:    {posts_path}")
    if comments_writer is not None:
        print(f"  comments: {comments_path}")
    if not args.no_trees and not args.no_comments:
        print(f"  trees:    {trees_dir}/S{{ss}}E{{ee}}.json")
    print(f"  summary:  {summary_path}")
    print(f"  log:      {scrape_log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
