"""
PullPush historical scraper for r/attackontitan (or any subreddit).

PullPush (https://pullpush.io) is the community-maintained successor to
Pushshift. It exposes Reddit's historical archive via a simple HTTP API and
does NOT require authentication, which makes it the right tool when you need
to go past Reddit's native ~1000-item listing cap.

Endpoints used:
    GET https://api.pullpush.io/reddit/search/submission/
    GET https://api.pullpush.io/reddit/search/comment/

Pagination strategy: walk backwards in time using the `before` parameter,
each page seeded with the oldest `created_utc` of the previous page.

Usage examples:
    # All-time posts archive for r/attackontitan (writes one CSV)
    python pullpush_scraper.py --kind posts

    # Posts AND comments for a date range
    python pullpush_scraper.py --kind both --after 2020-01-01 --before 2021-01-01

    # Just comments containing a keyword
    python pullpush_scraper.py --kind comments --query "ending"

    # Cap how far we walk (useful for pilot runs)
    python pullpush_scraper.py --kind posts --max-items 5000

Notes:
- PullPush is community-run; coverage of very recent days can lag and very
  old data can be patchy. Treat it as best-effort historical, and combine
  with the PRAW scraper for the live present-day window.
- Score / num_comments fields reflect values at the time PullPush ingested
  the item, not the current live values. For up-to-date scores re-fetch
  with PRAW using the post_id.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json

from tqdm import tqdm


PULLPUSH_BASE = "https://api.pullpush.io/reddit/search"
PAGE_SIZE = 100  # PullPush max per request
DEFAULT_UA = "aot-research-scraper/0.1 (pullpush historical archive)"


# ---------------------------------------------------------------------------
# Data shapes (intentionally close to the PRAW scraper's columns so the two
# datasets can be concatenated)
# ---------------------------------------------------------------------------


@dataclass
class PostRow:
    post_id: str
    subreddit: str
    title: str
    selftext: str
    author: str
    created_utc: float
    created_iso: str
    score: int
    num_comments: int
    permalink: str
    url: str
    domain: str
    is_self: bool
    over_18: bool
    spoiler: bool
    stickied: bool
    locked: bool
    link_flair_text: str
    distinguished: str
    fetched_iso: str
    source: str  # always "pullpush"


@dataclass
class CommentRow:
    comment_id: str
    post_id: str  # without t3_ prefix
    subreddit: str
    parent_id: str
    is_top_level: bool
    author: str
    body: str
    created_utc: float
    created_iso: str
    score: int
    is_submitter: bool
    stickied: bool
    distinguished: str
    permalink: str
    fetched_iso: str
    source: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_date(s: str) -> int:
    """Accept YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS, or a bare epoch int."""
    if s.isdigit():
        return int(s)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Bad date: {s!r}")


def fetch_page(kind: str, params: dict, user_agent: str,
               retries: int = 5, backoff: float = 2.0) -> list[dict]:
    """One GET to PullPush with retry on transient failures."""
    assert kind in {"submission", "comment"}
    url = f"{PULLPUSH_BASE}/{kind}/?{urlencode(params)}"
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": user_agent})
            with urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return payload.get("data", [])
        except HTTPError as e:
            last_err = e
            # 429 = rate limit, 5xx = transient
            if e.code in (429, 500, 502, 503, 504):
                sleep_for = backoff * (2 ** attempt)
                print(f"[pullpush] HTTP {e.code}, sleeping {sleep_for:.1f}s "
                      f"(attempt {attempt+1}/{retries})", file=sys.stderr)
                time.sleep(sleep_for)
                continue
            raise
        except URLError as e:
            last_err = e
            sleep_for = backoff * (2 ** attempt)
            print(f"[pullpush] network error {e}, sleeping {sleep_for:.1f}s "
                  f"(attempt {attempt+1}/{retries})", file=sys.stderr)
            time.sleep(sleep_for)
    raise RuntimeError(f"PullPush gave up after {retries} retries: {last_err}")


def iter_items(kind: str, subreddit: str, after: Optional[int],
               before: Optional[int], query: Optional[str],
               user_agent: str, sleep_between: float) -> Iterator[dict]:
    """Yield every item PullPush has, walking backwards in time via `before`."""
    cursor = before  # start at the upper bound (or open-ended if None)
    seen_ids: set[str] = set()

    while True:
        params = {
            "subreddit": subreddit,
            "size": PAGE_SIZE,
            "sort": "desc",
            "sort_type": "created_utc",
        }
        if cursor is not None:
            params["before"] = cursor
        if after is not None:
            params["after"] = after
        if query:
            params["q"] = query

        page = fetch_page(kind, params, user_agent=user_agent)
        if not page:
            return

        new_in_page = 0
        oldest_ts = None
        for item in page:
            iid = item.get("id")
            if not iid or iid in seen_ids:
                continue
            seen_ids.add(iid)
            new_in_page += 1
            ts = item.get("created_utc")
            if ts is not None and (oldest_ts is None or ts < oldest_ts):
                oldest_ts = ts
            yield item

        # PullPush may return duplicates near page boundaries; if we got
        # nothing new this page we're done.
        if new_in_page == 0 or oldest_ts is None:
            return

        # Move the window: next page should end *before* the oldest item we
        # just saw. Subtract 1s so we don't refetch that boundary item.
        next_cursor = int(oldest_ts) - 1
        if cursor is not None and next_cursor >= cursor:
            # Safety: cursor isn't advancing, bail to avoid infinite loop
            return
        cursor = next_cursor

        if sleep_between > 0:
            time.sleep(sleep_between)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def post_dict_to_row(d: dict) -> PostRow:
    permalink = d.get("permalink") or ""
    if permalink and not permalink.startswith("http"):
        permalink = f"https://www.reddit.com{permalink}"
    ts = float(d.get("created_utc") or 0)
    return PostRow(
        post_id=d.get("id") or "",
        subreddit=d.get("subreddit") or "",
        title=d.get("title") or "",
        selftext=d.get("selftext") or "",
        author=d.get("author") or "[deleted]",
        created_utc=ts,
        created_iso=_iso(ts) if ts else "",
        score=int(d.get("score") or 0),
        num_comments=int(d.get("num_comments") or 0),
        permalink=permalink,
        url=d.get("url") or "",
        domain=d.get("domain") or "",
        is_self=bool(d.get("is_self")),
        over_18=bool(d.get("over_18")),
        spoiler=bool(d.get("spoiler")),
        stickied=bool(d.get("stickied")),
        locked=bool(d.get("locked")),
        link_flair_text=d.get("link_flair_text") or "",
        distinguished=d.get("distinguished") or "",
        fetched_iso=_now_iso(),
        source="pullpush",
    )


def comment_dict_to_row(d: dict) -> CommentRow:
    parent_id = d.get("parent_id") or ""
    link_id = d.get("link_id") or ""
    # strip t3_ from link_id to get the post_id
    post_id = link_id[3:] if link_id.startswith("t3_") else link_id
    permalink = d.get("permalink") or ""
    if permalink and not permalink.startswith("http"):
        permalink = f"https://www.reddit.com{permalink}"
    ts = float(d.get("created_utc") or 0)
    return CommentRow(
        comment_id=d.get("id") or "",
        post_id=post_id,
        subreddit=d.get("subreddit") or "",
        parent_id=parent_id,
        is_top_level=parent_id.startswith("t3_"),
        author=d.get("author") or "[deleted]",
        body=d.get("body") or "",
        created_utc=ts,
        created_iso=_iso(ts) if ts else "",
        score=int(d.get("score") or 0),
        is_submitter=bool(d.get("is_submitter")),
        stickied=bool(d.get("stickied")),
        distinguished=d.get("distinguished") or "",
        permalink=permalink,
        fetched_iso=_now_iso(),
        source="pullpush",
    )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


class CsvWriter:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._writer = None

    def write(self, row) -> None:
        d = asdict(row)
        if self._writer is None:
            self._fh = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=list(d.keys()))
            self._writer.writeheader()
        self._writer.writerow(d)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Historical Reddit scraper using PullPush (Pushshift successor).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subreddit", default="attackontitan",
                   help="Subreddit name (without the r/)")
    p.add_argument("--kind", default="posts",
                   choices=["posts", "comments", "both"],
                   help="What to fetch")
    p.add_argument("--after", type=_parse_date, default=None,
                   help="Only items created at or after this date "
                        "(YYYY-MM-DD or epoch seconds)")
    p.add_argument("--before", type=_parse_date, default=None,
                   help="Only items created strictly before this date "
                        "(YYYY-MM-DD or epoch seconds)")
    p.add_argument("--query", default=None,
                   help="Full-text filter (PullPush 'q' parameter)")
    p.add_argument("--max-items", type=int, default=0,
                   help="Cap total items per kind (0 = no cap)")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds to sleep between pages (be polite)")
    p.add_argument("--output-dir", default="data",
                   help="Directory to write CSV files into")
    p.add_argument("--tag", default="",
                   help="Optional tag appended to output filenames")
    p.add_argument("--user-agent", default=DEFAULT_UA,
                   help="HTTP User-Agent string")
    return p.parse_args()


def scrape_one(kind: str, args, out_dir: Path, stamp: str, tag: str) -> tuple[Path, int]:
    api_kind = "submission" if kind == "posts" else "comment"
    label = "posts" if kind == "posts" else "comments"

    range_part = ""
    if args.after is not None or args.before is not None:
        a = datetime.fromtimestamp(args.after, tz=timezone.utc).strftime("%Y%m%d") if args.after else "open"
        b = datetime.fromtimestamp(args.before, tz=timezone.utc).strftime("%Y%m%d") if args.before else "open"
        range_part = f"_{a}-{b}"

    out_path = out_dir / f"pullpush_{label}_{args.subreddit}{range_part}_{stamp}{tag}.csv"
    print(f"[pullpush] → {out_path}")

    writer = CsvWriter(out_path)
    n = 0
    pbar = tqdm(unit=label, desc=label)
    try:
        for item in iter_items(
            kind=api_kind,
            subreddit=args.subreddit,
            after=args.after,
            before=args.before,
            query=args.query,
            user_agent=args.user_agent,
            sleep_between=args.sleep,
        ):
            row = (post_dict_to_row(item) if kind == "posts"
                   else comment_dict_to_row(item))
            writer.write(row)
            n += 1
            pbar.update(1)
            if args.max_items and n >= args.max_items:
                print(f"[pullpush] hit --max-items={args.max_items}, stopping")
                break
    finally:
        pbar.close()
        writer.close()

    return out_path, n


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""

    print(f"[pullpush] subreddit=r/{args.subreddit}  kind={args.kind}  "
          f"after={args.after}  before={args.before}  query={args.query!r}")

    kinds = ["posts", "comments"] if args.kind == "both" else [args.kind]
    summary: list[tuple[str, Path, int]] = []
    for k in kinds:
        path, n = scrape_one(k, args, out_dir, stamp, tag)
        summary.append((k, path, n))

    print("\n[pullpush] done.")
    for k, path, n in summary:
        print(f"  {k}: {n} rows → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
