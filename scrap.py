"""
Attack on Titan Transcript Scraper + Speaker Diarization
=========================================================
- 抓 Springfield 所有集數 transcript
- 合併跨行台詞
- 用 Claude 做 speaker diarization，並做 validation
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import anthropic
import json
import re
from pathlib import Path

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_URL  = "https://www.springfieldspringfield.co.uk"
SHOW_SLUG = "attack-on-titan-2013"
OUTPUT_CSV   = "aot_transcripts.csv"
RAW_DUMP_DIR = Path("raw_transcripts")   # 把每集 raw text 先存起來，方便 debug
RAW_DUMP_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": f"{BASE_URL}/episode_scripts.php?tv-show={SHOW_SLUG}",
}

client = anthropic.Anthropic()


# ─────────────────────────────────────────────
# Step 1: 取得集數清單
# ─────────────────────────────────────────────
def get_episode_list() -> list[dict]:
    url  = f"{BASE_URL}/episode_scripts.php?tv-show={SHOW_SLUG}"
    soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, "html.parser")

    episodes = []
    for a in soup.select('a[href*="view_episode_scripts"]'):
        href   = a["href"]
        ep_match = re.search(r"episode=(s(\d+)e(\d+))", href)
        if ep_match:
            episodes.append({
                "episode_id": ep_match.group(1),
                "season":     int(ep_match.group(2)),
                "episode":    int(ep_match.group(3)),
                "title":      a.text.strip(),
                "url":        BASE_URL + "/" + href,
            })
    return episodes


# ─────────────────────────────────────────────
# Step 2: 爬單集，並合併跨行台詞
# ─────────────────────────────────────────────
def scrape_and_merge(url: str, episode_id: str) -> list[str]:
    """
    Springfield 的 transcript 用 <br> 換行，
    跨行台詞的判斷邏輯：
      - 新 speaker 的行格式通常是 "Name: text" 或全大寫 / 特定 pattern
      - 不符合 speaker 開頭的行 → 接在前一行後面
    回傳：list of merged lines (每個元素是一段完整台詞/narration)
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")

    # 嘗試多個可能的 container selector
    container = (
        soup.find("div", class_="scrolling-script-container")
        or soup.find("div", class_="episode-script")
        or soup.find("div", id="script-text")
    )

    if not container:
        # Fallback: 找最長的 div
        all_divs = soup.find_all("div")
        container = max(all_divs, key=lambda d: len(d.get_text()), default=None)

    if not container:
        return []

    # 用 <br> 切割，保留原始行結構
    raw_lines = []
    for item in container.descendants:
        if isinstance(item, str):
            text = item.strip()
            if text:
                raw_lines.append(text)
        elif item.name == "br":
            raw_lines.append("__BR__")

    # 重建行（用 __BR__ 作分隔）
    lines = " ".join(raw_lines).split("__BR__")
    lines = [l.strip() for l in lines if l.strip()]

    # ── 合併跨行台詞 ──────────────────────────────
    # 判斷是否為「新段落開頭」的啟發式規則：
    #   1. 包含明確 speaker tag：  "Name: ..."
    #   2. 全部大寫（narration / 音效描述）
    #   3. 以 [ 或 ( 開頭（場景描述）
    #   4. 空的行 → 強制分段

    def is_new_segment(line: str) -> bool:
        if not line:
            return True
        # 有角色標記: "Eren:", "NARRATOR:", etc.
        if re.match(r'^[A-Z][A-Za-z\s\-\'\.]+:\s', line):
            return True
        # 全大寫短句（通常是 narration / 音效）
        if line.isupper() and len(line) < 80:
            return True
        # 場景描述
        if line.startswith(("[", "(")):
            return True
        return False

    merged = []
    buffer = ""
    for line in lines:
        if is_new_segment(line):
            if buffer:
                merged.append(buffer.strip())
            buffer = line
        else:
            # 跨行：接在 buffer 後面，用空格連接
            buffer = (buffer + " " + line).strip() if buffer else line

    if buffer:
        merged.append(buffer.strip())

    # 存 raw dump 方便 debug
    raw_path = RAW_DUMP_DIR / f"{episode_id}_raw.txt"
    raw_path.write_text("\n".join(merged), encoding="utf-8")

    return merged


# ─────────────────────────────────────────────
# Step 3: Claude speaker diarization
# ─────────────────────────────────────────────

DIARIZE_SYSTEM = """You are a precise dialogue analyst for the anime "Attack on Titan".

Given a list of transcript lines from one episode, your task is to label each line with its speaker.

RULES:
1. If the line already contains a speaker prefix like "Eren: I'll fight.", extract "Eren" as speaker and the rest as line.
2. If the line is narration (no clear speaker, describes events/setting), use speaker = "NARRATOR".
3. If the line is a sound effect or stage direction like "[Thunder]" or "(gasps)", use speaker = "SFX".
4. If the speaker is genuinely ambiguous, use speaker = "UNKNOWN".
5. Never invent or guess a speaker if there is no textual or strong contextual evidence.
6. For lines WITHOUT a speaker prefix, use context (surrounding lines, character knowledge) to infer.

OUTPUT FORMAT:
Return ONLY a JSON array. No markdown, no explanation. Example:
[
  {"speaker": "Eren", "line": "I'll kill every last one of them.", "confidence": "high"},
  {"speaker": "NARRATOR", "line": "Year 850. The wall has stood for 100 years.", "confidence": "high"},
  {"speaker": "UNKNOWN", "line": "We have to retreat now!", "confidence": "low"}
]

confidence values: "high" | "medium" | "low"
"""

def diarize_chunk(lines: list[str], episode_id: str, chunk_idx: int, total_chunks: int) -> list[dict]:
    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))

    prompt = f"""Episode: {episode_id} | Chunk {chunk_idx+1}/{total_chunks}

Lines to label:
{numbered}

Return a JSON array with exactly {len(lines)} objects (one per line, in order)."""

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=DIARIZE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
            result = json.loads(raw)

            # ── Validation ────────────────────────────────────
            # 確保回傳數量和輸入一樣多
            if len(result) != len(lines):
                print(f"    ⚠ Count mismatch (got {len(result)}, expected {len(lines)}), retrying...")
                time.sleep(2)
                continue

            # 確保每個 item 有必要欄位
            validated = []
            for i, item in enumerate(result):
                validated.append({
                    "speaker":    str(item.get("speaker", "UNKNOWN")).strip(),
                    "line":       str(item.get("line",    lines[i])).strip(),
                    "confidence": str(item.get("confidence", "low")).strip(),
                    "raw_input":  lines[i],   # 保留原始輸入，方便之後 QA
                })
            return validated

        except json.JSONDecodeError as e:
            print(f"    ✗ JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(3)
        except Exception as e:
            print(f"    ✗ API error (attempt {attempt+1}): {e}")
            time.sleep(5)

    # 全部 attempt 失敗 → fallback
    print(f"    ✗ All attempts failed, using UNKNOWN fallback")
    return [
        {"speaker": "UNKNOWN", "line": l, "confidence": "low", "raw_input": l}
        for l in lines
    ]


def diarize_episode(merged_lines: list[str], episode_id: str, chunk_size: int = 40) -> list[dict]:
    """把 merged lines 切塊送 Claude，回傳完整 diarization 結果"""
    chunks = [merged_lines[i:i+chunk_size] for i in range(0, len(merged_lines), chunk_size)]
    all_results = []

    for idx, chunk in enumerate(chunks):
        print(f"    Chunk {idx+1}/{len(chunks)} ({len(chunk)} lines)...")
        results = diarize_chunk(chunk, episode_id, idx, len(chunks))
        all_results.extend(results)
        time.sleep(1.0)   # 避免 rate limit

    return all_results


# ─────────────────────────────────────────────
# Step 4: 主流程
# ─────────────────────────────────────────────

def run(start_from: str = None, only_season: int = None):
    """
    start_from: episode_id (e.g. "s02e03"), 從這集開始跑（斷點續跑用）
    only_season: 只跑某一季
    """
    episodes = get_episode_list()
    print(f"Found {len(episodes)} episodes total")

    # 斷點續跑：如果 CSV 已存在，skip 已處理的集數
    done_ids = set()
    output_path = Path(OUTPUT_CSV)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        done_ids = set(existing["episode_id"].unique())
        print(f"Resuming: {len(done_ids)} episodes already done")

    all_rows = []
    skipping = bool(start_from)

    for ep in episodes:
        ep_id = ep["episode_id"]

        # 斷點
        if start_from and ep_id == start_from:
            skipping = False
        if skipping:
            continue

        # 季篩選
        if only_season and ep["season"] != only_season:
            continue

        # 已處理過
        if ep_id in done_ids:
            print(f"Skip {ep_id} (already done)")
            continue

        print(f"\n[{ep_id}] {ep['title']}")

        # 1. 爬 + 合併跨行
        merged = scrape_and_merge(ep["url"], ep_id)
        if not merged:
            print(f"  ✗ No transcript found")
            continue
        print(f"  {len(merged)} segments after merge")

        # 2. Diarize
        diarized = diarize_episode(merged, ep_id)

        # 3. 整理成 rows
        for d in diarized:
            all_rows.append({
                "season":      ep["season"],
                "episode":     ep["episode"],
                "episode_id":  ep_id,
                "title":       ep["title"],
                "speaker":     d["speaker"],
                "line":        d["line"],
                "confidence":  d["confidence"],
                "raw_input":   d["raw_input"],
            })

        # 4. 每集結束後就 append 到 CSV（避免中途 crash 損失進度）
        ep_df = pd.DataFrame(all_rows)
        if output_path.exists():
            ep_df.to_csv(output_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            ep_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        all_rows = []   # 清空 buffer

        print(f"  ✓ {len(diarized)} lines saved")
        time.sleep(2.5)  # Springfield rate limit

    print(f"\n✅ Done! Results in {OUTPUT_CSV}")


# ─────────────────────────────────────────────
# Step 5: 事後 QA — 找低信心 / UNKNOWN 的行
# ─────────────────────────────────────────────

def review_low_confidence(csv_path: str = OUTPUT_CSV):
    """印出 confidence=low 或 speaker=UNKNOWN 的行，方便人工 review"""
    df = pd.read_csv(csv_path)
    flagged = df[(df["confidence"] == "low") | (df["speaker"] == "UNKNOWN")]

    print(f"\n=== Low confidence / UNKNOWN: {len(flagged)} lines ===")
    for _, row in flagged.iterrows():
        print(f"[{row['episode_id']}] {row['speaker']} ({row['confidence']})")
        print(f"  raw:  {row['raw_input']}")
        print(f"  line: {row['line']}")
        print()

    # 存成獨立檔案方便 review
    flagged.to_csv("aot_review.csv", index=False, encoding="utf-8-sig")
    print("Saved to aot_review.csv")
    return flagged


def speaker_summary(csv_path: str = OUTPUT_CSV):
    """每個 speaker 的台詞數統計"""
    df = pd.read_csv(csv_path)
    summary = (
        df.groupby("speaker")["line"]
        .count()
        .sort_values(ascending=False)
        .rename("line_count")
    )
    print(summary.to_string())
    return summary


# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 跑全集
    run()

    # 事後 QA
    review_low_confidence()
    speaker_summary()