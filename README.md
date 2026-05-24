# Moral Disengagement and In/Out-Group Rhetoric in *Attack on Titan*

A computational text-analysis pipeline that extracts, labels, and scores all 88 episodes of the anime *Attack on Titan* for moral disengagement strategies and in/out-group rhetoric, using large language models and psychometric factor analysis.

---

## Overview

This project applies Bandura's (1999) moral disengagement framework and social identity theory to the dialogue of *Attack on Titan* (AoT). Each spoken line in the series is scored on eight moral disengagement dimensions and three in/out-group rhetoric dimensions by an LLM judge (GPT or Qwen), classified for emotion and sentiment by fine-tuned transformer models, and then reduced to latent factors via confirmatory factor analysis in R. The result is a character × episode panel dataset suitable for regression analysis.

A parallel corpus of Reddit audience comments is processed through the same emotion/sentiment pipeline for potential audience-response analyses.

---

## Repository Layout

```
SCISS/
├── src/
│   ├── scrapping.py          # Step 1 — scrape transcripts
│   ├── fetch_metadata.py     # Step 2 — fetch TVmaze episode metadata
│   ├── label_gemini.py       # Step 3 — Gemini 2.5 Pro character attribution
│   ├── label_gpt_mini.py     # Step 4a — GPT MD + in/out scoring
│   ├── label_qwen.py         # Step 4b — Qwen scoring (API)
│   ├── label_qwen_vllm.py    # Step 4c — Qwen2.5-32B scoring (vLLM / TACC)
│   └── classify.py           # Step 5 — emotion & sentiment (HuggingFace / TACC GPU)
│
├── prompt/
│   ├── moral_disengagement_system_prompt.txt   # 8-strategy MD rubric with few-shot examples
│   └── in_out_group_prompt.txt                 # 3-indicator in/out-group rubric
│
├── data/
│   ├── raw/
│   │   ├── aot_transcripts_raw.csv   # Scraped transcripts (no labels)
│   │   ├── aot_episode_metadata.csv  # TVmaze metadata
│   │   └── aot.csv                   # Merged transcripts + metadata
│   ├── aot_labeled.csv               # Gemini character labels (~20 500 lines)
│   ├── aot_labeled_filtered.csv      # Filtered subset sent to LLM scoring
│   ├── aot_gpt_mini_labeled.csv      # GPT MD + in/out scores (~8 400 lines)
│   ├── aot_qwen_labeled.csv          # Qwen MD + in/out scores
│   ├── aot_classified.csv            # Transcript emotion + sentiment scores
│   ├── aot_comments_classified.csv   # Reddit comment emotion + sentiment (~1.38 M rows)
│   └── AoT_gpt_data.csv              # Character × episode panel data (analysis-ready)
│
├── transcript/               # Per-episode plain-text transcripts (88 files)
│   ├── s01e01.txt
│   └── ...
│
├── figures/                  # Publication-ready figures (PNG, 300 dpi)
│   ├── fig1_top30_characters.png
│   ├── fig2_factor_scores_by_episode.png
│   ├── fig3a_md_factor_scores_by_character.png
│   ├── fig3b_inout_factor_scores_by_character.png
│   ├── fig4_line_length_distribution.png
│   ├── fig5_correlation_heatmap.png
│   ├── fig6_md_factor_loadings.png
│   └── fig7_scree_plot_parallel_analysis.png
│
├── tables/                   # LaTeX tables
│   ├── fa_md.tex             # MD factor loadings + variance explained
│   ├── fa_inout.tex          # In/out-group factor loadings
│   └── prompts.tex           # Prompt documentation table
│
├── main.R                    # Factor analysis, visualization, panel aggregation
├── main.ipynb                # Exploratory analysis notebook
├── run_classify.sh           # SLURM job script — emotion/sentiment (TACC GPU)
├── run_label.sh              # SLURM job script — Qwen2.5-32B scoring (TACC)
└── requirements.txt          # Python dependencies
```

---

## Pipeline

### Step 1 — Scrape Transcripts

```bash
python src/scrapping.py
```

Discovers all 88 episodes from the Springfield Springfield listing page and scrapes each `div.scrolling-script-container`. Each `NavigableString` between `<br>` tags becomes one CSV row.

**Outputs:**
- `data/raw/aot_transcripts_raw.csv` — columns: `sentence`, `episode`, `season`, `character` (blank)
- `transcript/s##e##.txt` — one plain-text file per episode

Progress is saved after every episode. The site is fully server-rendered; no browser automation is required.

---

### Step 2 — Fetch Episode Metadata

```bash
python src/fetch_metadata.py
```

Calls the free [TVmaze API](https://www.tvmaze.com/api) (no key required) and merges the result with the raw transcript CSV.

**Outputs:**
- `data/raw/aot_episode_metadata.csv` — columns: `season`, `episode`, `title`, `airdate`, `runtime`, `rating`
- `data/raw/aot.csv` — merged transcripts + metadata (input to Step 3)

---

### Step 3 — Character Attribution (Gemini 2.5 Pro Batch API)

```bash
pip install google-genai
GEMINI_API_KEY=your_key python src/label_gemini.py
```

Groups `data/raw/aot.csv` by `(season, episode)` and submits all 88 episodes as one Gemini Batch API job (~50 % cheaper than synchronous calls). Each line is assigned its speaker.

**Labels:**

| Label      | Meaning                                                              |
|------------|----------------------------------------------------------------------|
| Named character | Any character appearing in the show (e.g., `Eren Yeager`)   |
| `Narrator` | Formal third-person voiceover                                       |
| `Crowd`    | Indistinguishable crowd noise or many unnamed voices                |
| `Unknown`  | Speaker genuinely cannot be determined                              |

`Multiple` is not a valid label; when a scraped block spans several speakers, Gemini assigns it to the dominant speaker.

**Resume behavior:** The batch job name is saved to `data/batch_job_name.txt` on submission. Re-running after an interruption resumes the existing job without resubmitting.

**Output:** `data/aot_labeled.csv` (~20 500 rows)

---

### Step 4 — Rhetorical Scoring (LLM Judge)

Each line in `data/aot_labeled_filtered.csv` is scored on eleven dimensions by an LLM that is given the full rubric and few-shot examples from `prompt/`. Scores use a 1–5 Likert scale (1 = Absent, 5 = Very Strong).

#### Moral Disengagement (8 strategies — Bandura 1999)

| Strategy | Description |
|---|---|
| Moral Justification | Harmful acts framed as necessary for a higher purpose |
| Euphemistic Labeling | Violence described in sanitized or technical language |
| Advantageous Comparison | Own actions minimized by comparison with worse actions |
| Displacement of Responsibility | Agency attributed to orders or authority |
| Diffusion of Responsibility | Accountability diluted across a collective |
| Disregard / Distortion of Consequences | Harmful outcomes minimized or dismissed |
| Dehumanization | Targets portrayed as subhuman, animal-like, or impure |
| Attribution of Blame | Victims portrayed as deserving or provoking the harm |

#### In/Out-Group Rhetoric (3 indicators — Social Identity Theory)

| Indicator | Description |
|---|---|
| Boundary Marking | Linguistic construction of an us-versus-them distinction |
| Threat Framing | Out-group portrayed as existential or civilizational threat |
| Solidarity Appeal | Invocation of shared fate or collective mission among in-group |

#### Option A — GPT (primary)

```bash
OPENAI_API_KEY=your_key python src/label_gpt_mini.py \
    --input  data/aot_labeled_filtered.csv \
    --output data/aot_gpt_mini_labeled.csv \
    --workers 10
```

Uses `gpt-5-mini` with concurrent threads and row-level checkpointing. Both the MD and in/out-group prompts are sent in separate API calls per row. Resume-safe: re-running detects and skips already-scored rows.

**Output:** `data/aot_gpt_mini_labeled.csv` (~8 400 rows, 11 score columns)

#### Option B — Qwen2.5-32B-Instruct (TACC / GPU cluster)

```bash
# Submit via SLURM on TACC Lonestar6 (3× A100 GPUs, tensor parallelism)
sbatch run_label.sh
```

Runs `src/label_qwen_vllm.py` with the vLLM engine. Uses tensor parallelism across 2 GPUs (`--tp 2`) to fit the ~64 GB model. Same prompt rubric as Option A.

**Output:** `data/aot_qwen_labeled.csv`

---

### Step 5 — Emotion and Sentiment Classification

```bash
# Pre-download models on the login node (one-time, ~2 GB):
python src/classify.py --download-only --models-dir $WORK/hf_cache

# Submit GPU inference job:
sbatch run_classify.sh
```

Runs two HuggingFace pipelines on the transcript and comment datasets using a single A100 GPU on TACC.

| Task | Model | Labels |
|---|---|---|
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | anger, disgust, fear, joy, neutral, sadness, surprise |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | negative, neutral, positive |

Both pipelines return a top label and per-class probability scores for every row. Supports `--resume` to append results to an existing output file.

**Outputs:**
- `data/aot_classified.csv` — transcript lines with emotion + sentiment columns
- `data/aot_comments_classified.csv` — Reddit audience comments (~1.38 M rows)

---

### Step 6 — Factor Analysis and Visualization (R)

```r
source("main.R")
```

Runs in R and requires `tidyverse`, `psych`, `ggplot2`, `patchwork`, and `xtable`.

**Factor structure:**

| Scale | Factors | Rotation | Estimator |
|---|---|---|---|
| Moral Disengagement (8 items) | 2 | Oblimin (theory-driven; MD dimensions are correlated per Bandura 1999) | Maximum Likelihood |
| In/Out-Group (3 items) | 1 | None (just-identified) | Maximum Likelihood |

Factor scores are estimated via regression and aggregated to a **character × episode panel** (`data/AoT_gpt_data.csv`), weighted by number of lines, for downstream regression analysis.

**Figures produced** (`figures/`):

| File | Content |
|---|---|
| `fig1_top30_characters.png` | Top 30 characters by dialogue line count |
| `fig2_factor_scores_by_episode.png` | Mean factor scores across episodes, colored by season |
| `fig3a_md_factor_scores_by_character.png` | MD factor score distributions for top 20 characters (≥30 lines) |
| `fig3b_inout_factor_scores_by_character.png` | In/Out factor score distributions for the same characters |
| `fig4_line_length_distribution.png` | Distribution of transcript line lengths (characters per line) |
| `fig5_correlation_heatmap.png` | Correlation matrix of all 11 indicators |
| `fig6_md_factor_loadings.png` | MD factor loading heatmap (oblimin, ML) |
| `fig7_scree_plot_parallel_analysis.png` | Scree plot with parallel analysis for MD items |

**LaTeX tables produced** (`tables/`):
- `fa_md.tex` — MD factor loadings, communalities, variance explained, and factor inter-correlation
- `fa_inout.tex` — In/Out factor loadings and variance explained

---

## Dataset Schemas

### `data/raw/aot.csv` / `data/aot_labeled.csv`

| Column | Type | Description |
|---|---|---|
| `season` | int | Season number (1–4) |
| `episode` | int | Episode number within season |
| `title` | str | Episode title |
| `airdate` | str | Original air date (YYYY-MM-DD) |
| `runtime` | int | Runtime in minutes |
| `rating` | float | TVmaze average audience rating |
| `character` | str | Speaker (blank in raw; filled by Gemini in labeled) |
| `sentence` | str | One dialogue line |

### `data/aot_gpt_mini_labeled.csv` (adds to above)

| Column | Type | Description |
|---|---|---|
| `md_parse_ok` | bool | MD JSON parsed successfully |
| `inout_parse_ok` | bool | In/out-group JSON parsed successfully |
| `md_moral_justification_score` | float | 1–5 |
| `md_euphemistic_labeling_score` | float | 1–5 |
| `md_advantageous_comparison_score` | float | 1–5 |
| `md_displacement_of_responsibility_score` | float | 1–5 |
| `md_diffusion_of_responsibility_score` | float | 1–5 |
| `md_disregard___distortion_of_consequences_score` | float | 1–5 |
| `md_dehumanization_score` | float | 1–5 |
| `md_attribution_of_blame_score` | float | 1–5 |
| `inout_boundary_marking_score` | float | 1–5 |
| `inout_threat_framing_score` | float | 1–5 |
| `inout_solidarity_appeal_score` | float | 1–5 |

### `data/aot_classified.csv` (adds to labeled)

| Column | Type | Description |
|---|---|---|
| `emotion_label` | str | Top predicted emotion |
| `emotion_{anger\|disgust\|fear\|joy\|neutral\|sadness\|surprise}` | float | Per-class probability |
| `sentiment_label` | str | Top predicted sentiment |
| `sentiment_{negative\|neutral\|positive}` | float | Per-class probability |

### `data/AoT_gpt_data.csv` (panel data)

| Column | Type | Description |
|---|---|---|
| `character` | str | Character name |
| `season`, `episode`, `title`, `airdate` | — | Episode identifiers |
| `n_lines` | int | Number of lines contributed by this character in this episode |
| `md_F1_mean`, `md_F2_mean` | float | Episode-level mean MD factor scores |
| `inout_F1_mean` | float | Episode-level mean in/out-group factor score |
| `ep_order` | int | Sequential episode index across all seasons |
| `season_f` | str | Season label (e.g., `Season 1`) |

---

## Requirements

**Python** (see `requirements.txt`):

```
requests
beautifulsoup4
lxml
pandas
python-dotenv
google-genai          # label_gemini.py
openai                # label_gpt_mini.py
torch
transformers
tqdm                  # classify.py
vllm                  # label_qwen_vllm.py (GPU cluster only)
```

**R packages:** `tidyverse`, `psych`, `ggplot2`, `patchwork`, `xtable`

Conda environment: **llm**

---

## Compute Notes

Emotion/sentiment classification and Qwen2.5-32B scoring are run on [TACC](https://www.tacc.utexas.edu/) GPU nodes via SLURM. The provided `run_classify.sh` and `run_label.sh` scripts are configured for TACC Lonestar6 (`gpu-a100` partition). Adjust `--account` and `--partition` before submitting.

GPT scoring (`label_gpt_mini.py`) runs locally or on any machine with internet access; concurrent threads (`--workers`) are throttled by your OpenAI rate limits.

---

## References

Bandura, A. (1999). Moral disengagement in the perpetration of inhumanities. *Personality and Social Psychology Review*, 3(3), 193–209.

Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup conflict. In W. G. Austin & S. Worchel (Eds.), *The social psychology of intergroup relations* (pp. 33–47). Brooks/Cole.
