# Attack on Titan Transcript Dataset

Scrape, store, and label all 88 Attack on Titan episode transcripts from [springfieldspringfield.co.uk](https://www.springfieldspringfield.co.uk), then annotate each dialogue line with a character name using the Gemini 2.5 Pro Batch API.

---

## Repository Layout

```
SCISS/
├── scrapping.py          # Step 1 — scrape transcripts → CSV + per-episode txt files
├── fetch_metadata.py     # Step 2 — fetch episode metadata from TVmaze API
├── label_gemini.py       # Step 3 — label characters via Gemini 2.5 Pro Batch API
├── main.ipynb            # Exploration / analysis notebook
│
├── data/
│   ├── aot_transcripts_raw.csv   # Output of scrapping.py (no character labels)
│   ├── aot_episode_metadata.csv  # Output of fetch_metadata.py
│   ├── aot.csv                   # Merged transcripts + metadata (input to label_gemini.py)
│   ├── aot_labeled.csv           # Output of label_gemini.py (with character labels)
│   └── batch_job_name.txt        # Saved Gemini batch job ID (created during labeling)
│
└── transcript/           # Per-episode plain-text transcripts
    ├── s01e01.txt
    ├── s01e02.txt
    └── ...
```

---

## Pipeline

### Step 1 — Scrape transcripts

```bash
python scrapping.py
```

- Discovers all 88 episodes from the Springfield Springfield listing page.
- Scrapes each episode's `div.scrolling-script-container`; each `NavigableString` between `<br>` tags becomes one CSV row.
- Outputs:
  - `data/aot_transcripts_raw.csv` — columns: `sentence`, `episode`, `season`, `character` (blank)
  - `transcript/s##e##.txt` — one file per episode, one line per dialogue line
- Saves progress after every episode so a crash doesn't lose work.
- Politely waits 2.5 s between requests. No browser automation needed — the site is fully server-rendered.

### Step 2 — Fetch episode metadata

```bash
python fetch_metadata.py
```

- Calls the free [TVmaze API](https://www.tvmaze.com/api) (no key required).
- Outputs `data/aot_episode_metadata.csv` — columns: `season`, `episode`, `title`, `airdate`, `runtime`, `rating`.
- Merge with the raw transcript CSV to produce `data/aot.csv` (the input for Step 3).

### Step 3 — Label characters

```bash
pip install google-genai
GEMINI_API_KEY=your_key python label_gemini.py
```

- Reads `data/aot.csv`, groups by `(season, episode)`.
- Builds one prompt per episode (all lines numbered) and submits all 88 as a single Gemini Batch API job — ~50% cheaper than real-time calls.
- Polls the job every 60 s until complete, then parses each episode's JSON response and writes `data/aot_labeled.csv`.
- **Resume-safe**: the job name is saved to `data/batch_job_name.txt` on submit. If the script is interrupted during polling, re-run it — it resumes the existing job instead of resubmitting.

---

## Output Schema

### `data/aot.csv` / `data/aot_labeled.csv`

| Column      | Type    | Description                                        |
|-------------|---------|----------------------------------------------------|
| `season`    | int     | Season number                                      |
| `episode`   | int     | Episode number within its season                   |
| `title`     | str     | Episode title                                      |
| `airdate`   | str     | Original air date (YYYY-MM-DD)                     |
| `runtime`   | int     | Runtime in minutes                                 |
| `rating`    | float   | TVmaze average rating                              |
| `character` | str     | Speaker name (blank in `aot.csv`; filled in `aot_labeled.csv`) |
| `sentence`  | str     | One dialogue line from the transcript              |

### `data/aot_transcripts_raw.csv`

| Column      | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| `sentence`  | str     | One dialogue line from the transcript            |
| `episode`   | int     | Episode number within its season                 |
| `season`    | int     | Season number                                    |
| `character` | str     | Always blank                                     |

---

## Character Labels

Gemini assigns each line one of the named characters from the show, or a special label:

| Label      | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| `Narrator` | Formal third-person voiceover describing the world or events            |
| `Crowd`    | Indistinguishable crowd noise, battle shouts, or many unnamed voices    |
| `Unknown`  | Speaker genuinely cannot be determined                                  |

`Multiple` is not a valid label. When a scraped block contains several characters' lines, Gemini assigns it to the dominant speaker.

---

## Requirements

```
requests
beautifulsoup4
lxml
pandas
python-dotenv
google-genai        # label_gemini.py (Batch API)
```

Conda environment: **llm**

---

## Utilities

### `main.ipynb`

Exploration and analysis notebook for the labeled dataset.
