# Attack on Titan Transcript Dataset

Scrape, store, and label all 88 Attack on Titan episode transcripts from [springfieldspringfield.co.uk](https://www.springfieldspringfield.co.uk), then annotate each dialogue line with a character name using Gemini 2.5 Pro.

---

## Repository Layout

```
SCISS/
├── scrapping.py          # Step 1 — scrape transcripts → CSV + per-episode txt files
├── fetch_metadata.py     # (optional) fetch episode metadata from TVmaze API
├── label_gemini.py       # Step 2 — label characters via Gemini 2.5 Pro
├── gen_txt.py            # Utility — regenerate txt files from an existing CSV
│
├── aot_transcripts_raw.csv      # Output of scrapping.py (no character labels)
├── aot_transcripts_labeled.csv  # Output of label_gemini.py (with character labels)
├── aot_episode_metadata.csv     # Output of fetch_metadata.py
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
  - `aot_transcripts_raw.csv` — columns: `sentence`, `episode`, `season`, `character` (blank)
  - `transcript/s##e##.txt` — one file per episode, one line per dialogue line
- Saves progress after every episode so a crash doesn't lose work.
- Politely waits 2.5 s between requests.

### Step 2 — Label characters

```bash
pip install google-generativeai
GEMINI_API_KEY=your_key python label_gemini.py
```

- Reads `aot_transcripts_raw.csv`, groups by `(season, episode)`.
- Sends batches of 30 lines to Gemini 2.5 Pro with a character roster prompt.
- Writes `aot_transcripts_labeled.csv` — same schema plus a filled `character` column.
- Saves after each episode; safe to resume if interrupted.

### (Optional) Fetch episode metadata

```bash
python fetch_metadata.py
```

- Calls the free [TVmaze API](https://www.tvmaze.com/api) (no key required).
- Outputs `aot_episode_metadata.csv` — columns: `season`, `episode`, `title`, `airdate`, `runtime`, `rating`, `summary`, `tvmaze_id`.

---

## Output Schema

### `aot_transcripts_raw.csv` / `aot_transcripts_labeled.csv`

| Column      | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| `sentence`  | str     | One dialogue line from the transcript            |
| `episode`   | int     | Episode number within its season                 |
| `season`    | int     | Season number                                    |
| `character` | str     | Speaker name (blank in raw; filled in labeled)   |

### `aot_episode_metadata.csv`

| Column      | Type    | Description                        |
|-------------|---------|------------------------------------|
| `season`    | int     | Season number                      |
| `episode`   | int     | Episode number                     |
| `title`     | str     | Episode title                      |
| `airdate`   | str     | Original air date (YYYY-MM-DD)     |
| `runtime`   | int     | Runtime in minutes                 |
| `rating`    | float   | TVmaze average rating              |
| `summary`   | str     | Plain-text episode summary         |
| `tvmaze_id` | int     | TVmaze episode ID                  |

---

## Character Labels

Gemini assigns each line one of the known characters or a special label:

| Label        | Meaning                                          |
|--------------|--------------------------------------------------|
| `Narrator`   | Voiceover / narration with no identifiable speaker |
| `Multiple`   | Line contains more than one distinct speaker     |
| `crowd line` | Indistinguishable crowd shout or battle noise    |
| `Unknown`    | Speaker genuinely cannot be determined           |

---

## Requirements

```
requests
beautifulsoup4
lxml
pandas
google-generativeai   # only for label_gemini.py
```

Conda environment: **llm**

No browser automation needed — Springfield Springfield is fully server-rendered.

---

## Utilities

### `gen_txt.py`

If you already have `aot_transcripts_raw.csv` but the `transcript/` folder is missing or incomplete, run:

```bash
python gen_txt.py
```

Reads the CSV and writes one `s##e##.txt` per episode into `transcript/`.
