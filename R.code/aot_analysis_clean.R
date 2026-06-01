# ============================================================
# Attack on Titan Reddit Comment Study
# Main analysis and robustness check
# ============================================================
# Purpose:
#   This script estimates the main models using GPT-mini coded dialogue data
#   and repeats the same analysis using Qwen coded dialogue data as a
#   robustness check.
#
# Required input files:
#   data/aot_comments_classified.csv
#   data/aot_gpt_mini_labeled.csv
#   data/aot_qwen_labeled.csv
#
# Output:
#   output/tables/main_results.txt
#   output/tables/robustness_results.txt
#   output/tables/qwen_main_results.txt
#   output/tables/qwen_robustness_results.txt
# ============================================================

# ---- 0. Setup ---------------------------------------------------------------

rm(list = ls())

packages <- c(
  "tidyverse",
  "stringr",
  "fixest"
)

invisible(lapply(packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' is required but not installed.", pkg))
  }
  library(pkg, character.only = TRUE)
}))

# Use relative paths so the repository runs on other computers.
data_dir <- "data"
output_dir <- file.path("output", "tables")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 1. Load data -----------------------------------------------------------

comments <- read.csv(file.path(data_dir, "aot_comments_classified.csv"))
mini <- read.csv(file.path(data_dir, "aot_gpt_mini_labeled.csv"))
qwen <- read.csv(file.path(data_dir, "aot_qwen_labeled.csv"))

# ---- 2. Character coding ----------------------------------------------------

character_map <- tibble::tibble(
  full_name = c(
    "eren yeager", "armin arlert", "mikasa ackerman", "reiner braun",
    "zeke yeager", "levi ackerman", "floch forster", "historia reiss",
    "jean kirstein", "gabi braun", "falco grice", "annie leonhart",
    "pieck finger", "ymir"
  ),
  character_key = c(
    "eren", "armin", "mikasa", "reiner", "zeke", "levi", "floch",
    "historia", "jean", "gabi", "falco", "annie", "pieck", "ymir"
  )
)

protagonists <- c("eren", "armin", "mikasa", "levi", "jean", "historia", "floch")
antagonists <- c("reiner", "zeke", "gabi", "falco", "annie", "pieck", "ymir")

# ---- 3. Helper functions ----------------------------------------------------

make_dialogue_data <- function(dialogue_data, character_map) {
  dialogue_data %>%
    filter(stringr::str_count(sentence, "\\S+") >= 5) %>%
    mutate(character_lower = stringr::str_to_lower(character)) %>%
    left_join(character_map, by = c("character_lower" = "full_name")) %>%
    filter(!is.na(character_key)) %>%
    group_by(character = character_key, season, episode) %>%
    summarise(
      dialogue_lines = n(),
      threat_framing = mean(inout_threat_framing_score, na.rm = TRUE),
      boundary_marking = mean(inout_boundary_marking_score, na.rm = TRUE),
      solidarity_appeal = mean(inout_solidarity_appeal_score, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      inout_composite = rowMeans(
        cbind(threat_framing, boundary_marking, solidarity_appeal),
        na.rm = TRUE
      )
    )
}

# NOTE:
# The original working script used an object called `character_episode_comments`.
# This helper assumes that the comment-level file has already been collapsed to
# the character-episode level or contains the columns used below. If your comments
# file is still comment-level, create the same character-episode summary before
# running this script.
#
# Required columns for `character_episode_comments`:
#   character, ep_season, ep_episode, n_comments, mean_anger, mean_disgust,
#   sentiment_polarization, mean_fear, mean_positive
make_comment_data <- function(comments) {
  required_cols <- c(
    "character", "ep_season", "ep_episode", "n_comments", "mean_anger",
    "mean_disgust", "sentiment_polarization", "mean_fear", "mean_positive"
  )

  missing_cols <- setdiff(required_cols, names(comments))
  if (length(missing_cols) > 0) {
    stop(
      paste0(
        "The comments file is missing required character-episode columns: ",
        paste(missing_cols, collapse = ", "),
        "\nIf the comments file is comment-level, first aggregate it to the ",
        "character-episode level."
      )
    )
  }

  comments %>%
    select(all_of(required_cols)) %>%
    distinct()
}

make_analysis_data <- function(dialogue_data, comments, character_map) {
  character_episode_dialogue <- make_dialogue_data(dialogue_data, character_map)
  character_episode_comments <- make_comment_data(comments)

  character_episode_dialogue %>%
    left_join(
      character_episode_comments,
      by = c(
        "character" = "character",
        "season" = "ep_season",
        "episode" = "ep_episode"
      )
    ) %>%
    mutate(
      char_type = case_when(
        character %in% protagonists ~ "protagonist",
        character %in% antagonists ~ "antagonist",
        TRUE ~ NA_character_
      ),
      char_type = factor(char_type, levels = c("antagonist", "protagonist")),
      threat_framing_s = as.numeric(scale(threat_framing)),
      boundary_marking_s = as.numeric(scale(boundary_marking)),
      solidarity_appeal_s = as.numeric(scale(solidarity_appeal)),
      inout_composite_s = as.numeric(scale(inout_composite)),
      dialogue_lines_s = as.numeric(scale(dialogue_lines))
    ) %>%
    filter(!is.na(n_comments), n_comments >= 10)
}

estimate_models <- function(analysis_data) {
  list(
    main = list(
      h1a_anger = feols(
        mean_anger ~ inout_composite_s + dialogue_lines_s | character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h1a_disgust = feols(
        mean_disgust ~ inout_composite_s + dialogue_lines_s | character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h1b_polarization = feols(
        sentiment_polarization ~ inout_composite_s + dialogue_lines_s | character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h1c_fear = feols(
        mean_fear ~ threat_framing_s + dialogue_lines_s | character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h2a_anger = feols(
        mean_anger ~ threat_framing_s + threat_framing_s:char_type + dialogue_lines_s |
          character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h2a_disgust = feols(
        mean_disgust ~ threat_framing_s + threat_framing_s:char_type + dialogue_lines_s |
          character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h2b_positive = feols(
        mean_positive ~ solidarity_appeal_s + solidarity_appeal_s:char_type + dialogue_lines_s |
          character + season,
        data = analysis_data,
        cluster = ~character
      )
    ),
    robust = list(
      h1c_robust = feols(
        mean_fear ~ threat_framing_s + boundary_marking_s + solidarity_appeal_s +
          dialogue_lines_s | character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h2a_anger_robust = feols(
        mean_anger ~ threat_framing_s + threat_framing_s:char_type +
          boundary_marking_s + solidarity_appeal_s + dialogue_lines_s |
          character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h2a_disgust_robust = feols(
        mean_disgust ~ threat_framing_s + threat_framing_s:char_type +
          boundary_marking_s + solidarity_appeal_s + dialogue_lines_s |
          character + season,
        data = analysis_data,
        cluster = ~character
      ),
      h2b_positive_robust = feols(
        mean_positive ~ solidarity_appeal_s + solidarity_appeal_s:char_type +
          threat_framing_s + boundary_marking_s + dialogue_lines_s |
          character + season,
        data = analysis_data,
        cluster = ~character
      )
    )
  )
}

save_etable <- function(models, file_name, headers) {
  table_text <- capture.output(
    etable(
      models,
      headers = headers,
      se.below = TRUE,
      digits = 3
    )
  )
  writeLines(table_text, file.path(output_dir, file_name))
}

# ---- 4. Main analysis: GPT-mini dialogue coding -----------------------------

analysis_mini <- make_analysis_data(mini, comments, character_map)
models_mini <- estimate_models(analysis_mini)

save_etable(
  models_mini$main,
  "main_results.txt",
  c(
    "H1a Anger", "H1a Disgust", "H1b Polarization", "H1c Fear",
    "H2a Anger", "H2a Disgust", "H2b Positive"
  )
)

save_etable(
  models_mini$robust,
  "robustness_results.txt",
  c("H1c Robust", "H2a Anger Robust", "H2a Disgust Robust", "H2b Robust")
)

# ---- 5. Robustness check: Qwen dialogue coding ------------------------------

analysis_qwen <- make_analysis_data(qwen, comments, character_map)
models_qwen <- estimate_models(analysis_qwen)

save_etable(
  models_qwen$main,
  "qwen_main_results.txt",
  c(
    "H1a Anger", "H1a Disgust", "H1b Polarization", "H1c Fear",
    "H2a Anger", "H2a Disgust", "H2b Positive"
  )
)

save_etable(
  models_qwen$robust,
  "qwen_robustness_results.txt",
  c("H1c Robust", "H2a Anger Robust", "H2a Disgust Robust", "H2b Robust")
)

# ---- 6. Diagnostics ---------------------------------------------------------

cat("Main analysis observations:", nrow(analysis_mini), "\n")
cat("Qwen robustness observations:", nrow(analysis_qwen), "\n")
cat("Tables saved to:", output_dir, "\n")
