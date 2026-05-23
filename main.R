### 05.23.2026
### Factor analysis of AoT GPT-mini labeled data
### MD (8 indicators) + In/Out (3 indicators) FA; character/episode aggregation

suppressPackageStartupMessages({
    library(tidyverse)
    library(psych)
    library(ggplot2)
    library(patchwork)
    library(forcats)
})

if (!requireNamespace("ggcorrplot", quietly = TRUE)) {
    install.packages("ggcorrplot", repos = "https://cloud.r-project.org")
}

dir.create("./figures", showWarnings = FALSE)

# ── Palette & theme (consistent across all figures; max 3 colors per plot) ────
col_primary   <- "#1F4068"
col_secondary <- "#E07B39"
col_neutral   <- "#A8B9C8"

season_pal <- c(
    "Season 1" = "#A8C4D9",
    "Season 2" = "#5B90B7",
    "Season 3" = "#2B6A96",
    "Season 4" = "#1F4068"
)

theme_journal <- function() {
    theme_minimal(base_size = 11) +
    theme(
        panel.grid.minor  = element_blank(),
        panel.grid.major  = element_line(color = "grey92", linewidth = 0.35),
        axis.line         = element_line(color = "grey35", linewidth = 0.4),
        axis.ticks        = element_line(color = "grey35", linewidth = 0.4),
        strip.text        = element_text(size = 9.5, face = "bold"),
        legend.position   = "bottom",
        legend.key.size   = unit(0.45, "cm"),
        legend.text       = element_text(size = 9),
        axis.text         = element_text(size = 9),
        axis.title        = element_text(size = 10),
        plot.margin       = margin(6, 10, 6, 6)
    )
}

# ── Variable definitions ───────────────────────────────────────────────────────
md_cols <- c(
    "md_moral_justification_score",
    "md_euphemistic_labeling_score",
    "md_advantageous_comparison_score",
    "md_displacement_of_responsibility_score",
    "md_diffusion_of_responsibility_score",
    "md_disregard___distortion_of_consequences_score",
    "md_dehumanization_score",
    "md_attribution_of_blame_score"
)

inout_cols <- c(
    "inout_boundary_marking_score",
    "inout_threat_framing_score",
    "inout_solidarity_appeal_score"
)

short_labels <- c(
    md_moral_justification_score                      = "Moral Just.",
    md_euphemistic_labeling_score                     = "Euphemistic",
    md_advantageous_comparison_score                  = "Adv. Comp.",
    md_displacement_of_responsibility_score           = "Displace.",
    md_diffusion_of_responsibility_score              = "Diffusion",
    `md_disregard___distortion_of_consequences_score` = "Disregard",
    md_dehumanization_score                           = "Dehumanize",
    md_attribution_of_blame_score                     = "Attr. Blame",
    inout_boundary_marking_score                      = "Boundary",
    inout_threat_framing_score                        = "Threat",
    inout_solidarity_appeal_score                     = "Solidarity"
)

# ── Load datasets ─────────────────────────────────────────────────────────────
df_gpt      <- read_csv("./data/aot_gpt_mini_labeled.csv",    show_col_types = FALSE)
df_qwen     <- read_csv("./data/aot_qwen_labeled.csv",        show_col_types = FALSE)
df_comments <- read_csv("./data/aot_comments_classified.csv", show_col_types = FALSE)

# ── Filter to successfully parsed rows with no NA scores ──────────────────────
df_clean <- df_gpt %>%
    filter(md_parse_ok == TRUE, inout_parse_ok == TRUE) %>%
    filter(if_all(all_of(c(md_cols, inout_cols)), ~ !is.na(.)))

cat(sprintf("Rows after filtering: %d of %d (dropped %d)\n",
            nrow(df_clean), nrow(df_gpt), nrow(df_gpt) - nrow(df_clean)))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Factor Analysis
# ═══════════════════════════════════════════════════════════════════════════════

cat("\n====== KMO + Bartlett (required for journal reporting) ======\n")
cat("\n--- KMO (MD indicators) ---\n")
print(KMO(df_clean[md_cols]))
cat("\n--- Bartlett's test (MD indicators) ---\n")
print(cortest.bartlett(cor(df_clean[md_cols]), n = nrow(df_clean)))
cat("\n--- KMO (In/Out indicators) ---\n")
print(KMO(df_clean[inout_cols]))
cat("\n--- Bartlett's test (In/Out indicators) ---\n")
print(cortest.bartlett(cor(df_clean[inout_cols]), n = nrow(df_clean)))

# ── Parallel analysis (retained for scree plot only; nfactors fixed below) ───
set.seed(42)
pa_md <- fa.parallel(df_clean[md_cols], fm = "ml", fa = "fa", plot = FALSE)
cat(sprintf("\nParallel analysis suggests %d factor(s); using 2 (theory-driven)\n",
            pa_md$nfact))

# ── MD FA: 8 indicators, oblique rotation ─────────────────────────────────────
# oblimin: MD dimensions are correlated per Bandura's (1986) theoretical model
# fm = "ml": maximum likelihood; defensible at N = 3895 with continuous Likert
fa_md <- fa(
    df_clean[md_cols],
    nfactors = 2,
    rotate   = "oblimin",
    fm       = "ml",
    scores   = "regression"
)
cat("\n====== MD Factor Loadings ======\n")
print(fa_md$loadings, cutoff = 0.3)
cat(sprintf("MD cumulative variance explained: %.1f%%\n",
            sum(fa_md$Vaccounted[2, ]) * 100))

# ── In/Out FA: 3 indicators → 1 factor (just-identified; 0 df) ───────────────
fa_inout <- fa(
    df_clean[inout_cols],
    nfactors = 1,
    rotate   = "none",
    fm       = "ml",
    scores   = "regression"
)
cat("\n====== In/Out Factor Loadings ======\n")
print(fa_inout$loadings)
cat(sprintf("In/Out variance explained: %.1f%%\n",
            fa_inout$Vaccounted["Proportion Var", 1] * 100))

# ── LaTeX tables for FA results ───────────────────────────────────────────────
library(xtable)
dir.create("./tables", showWarnings = FALSE)

# Suppress loadings below cutoff (replace with blank for cleaner table)
fmt_load <- function(x, cutoff = 0.30) {
    ifelse(abs(x) < cutoff, "", sprintf("%.2f", x))
}

# MD FA table — loadings, communalities, variance explained, factor correlation
md_load <- as.data.frame(unclass(fa_md$loadings))

md_body <- data.frame(
    Indicator  = unname(short_labels[md_cols]),
    `Factor 1` = fmt_load(md_load[, 1]),
    `Factor 2` = fmt_load(md_load[, 2]),
    `$h^2$`    = sprintf("%.2f", fa_md$communality),
    check.names = FALSE
)

md_footer <- data.frame(
    Indicator  = c("\\% Variance", "Cumulative \\%"),
    `Factor 1` = c(
        sprintf("%.1f", fa_md$Vaccounted["Proportion Var", 1] * 100),
        sprintf("%.1f", fa_md$Vaccounted["Cumulative Var",  1] * 100)
    ),
    `Factor 2` = c(
        sprintf("%.1f", fa_md$Vaccounted["Proportion Var", 2] * 100),
        sprintf("%.1f", fa_md$Vaccounted["Cumulative Var",  2] * 100)
    ),
    `$h^2$`    = c("", ""),
    check.names = FALSE
)

md_full <- rbind(md_body, md_footer)

phi_note <- if (!is.null(fa_md$Phi)) {
    sprintf("Note: Factor inter-correlation $r = %.2f$.", fa_md$Phi[1, 2])
} else ""

xt_md <- xtable(
    md_full,
    caption = paste0(
        "Moral Disengagement Factor Loadings (oblimin rotation, ML; $N = 3{,}895$). ",
        "Loadings $< .30$ are suppressed. ", phi_note
    ),
    label = "tab:fa_md"
)
align(xt_md) <- c("l", "l", "r", "r", "r")

print(xt_md,
      file                   = "./tables/fa_md.tex",
      include.rownames       = FALSE,
      booktabs               = TRUE,
      caption.placement      = "top",
      comment                = FALSE,
      sanitize.text.function = identity,
      add.to.row             = list(
          pos     = list(nrow(md_body)),
          command = "\\midrule\n"
      ))

# In/Out FA table — all loadings shown (only 3 items; cutoff = 0)
io_load <- as.data.frame(unclass(fa_inout$loadings))

io_body <- data.frame(
    Indicator  = unname(short_labels[inout_cols]),
    `Factor 1` = sprintf("%.2f", io_load[, 1]),
    `$h^2$`    = sprintf("%.2f", fa_inout$communality),
    check.names = FALSE
)

io_footer <- data.frame(
    Indicator  = "\\% Variance",
    `Factor 1` = sprintf("%.1f", fa_inout$Vaccounted["Proportion Var", 1] * 100),
    `$h^2$`    = "",
    check.names = FALSE
)

io_full <- rbind(io_body, io_footer)

xt_io <- xtable(
    io_full,
    caption = "In/Out-Group Factor Loadings (1 factor, ML; $N = 3{,}895$).",
    label   = "tab:fa_inout"
)
align(xt_io) <- c("l", "l", "r", "r")

print(xt_io,
      file                   = "./tables/fa_inout.tex",
      include.rownames       = FALSE,
      booktabs               = TRUE,
      caption.placement      = "top",
      comment                = FALSE,
      sanitize.text.function = identity,
      add.to.row             = list(
          pos     = list(nrow(io_body)),
          command = "\\midrule\n"
      ))

cat("LaTeX tables written to ./tables/\n")

# ── Attach factor scores ──────────────────────────────────────────────────────
md_score_names    <- c("md_F1", "md_F2")
inout_score_names <- "inout_F1"
all_score_names   <- c(md_score_names, inout_score_names)

md_scores_df    <- as.data.frame(fa_md$scores)
inout_scores_df <- as.data.frame(fa_inout$scores)
names(md_scores_df)    <- md_score_names
names(inout_scores_df) <- inout_score_names

df_scored <- bind_cols(df_clean, md_scores_df, inout_scores_df)

# Human-readable factor labels for plot facets
factor_labels <- c(md_F1 = "MD Factor 1", md_F2 = "MD Factor 2", inout_F1 = "In/Out Factor")

# ── Aggregate by character × episode (panel data for regression) ─────────────
agg_panel <- df_scored %>%
    group_by(character, season, episode, title, airdate, rating) %>%
    summarise(
        n_lines = n(),
        across(all_of(all_score_names), ~ mean(., na.rm = TRUE), .names = "{.col}_mean"),
        .groups = "drop"
    ) %>%
    arrange(character, season, episode) %>%
    mutate(
        ep_order = match(paste0(season, "_", episode),
                         unique(paste0(season, "_", episode))),
        season_f = factor(paste0("Season ", season), levels = paste0("Season ", 1:4))
    )

write_csv(agg_panel, "./data/AoT_gpt_data.csv")
cat(sprintf("\nPanel dataset: %d rows (%d character-episode observations) written to ./data/agg_panel_scores.csv\n",
            nrow(agg_panel), nrow(agg_panel)))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

save_fig <- function(p, name, w, h) {
    # ggsave(paste0("./figures/", name, ".pdf"), p, width = w, height = h)
    ggsave(paste0("./figures/", name, ".png"), p, width = w, height = h, dpi = 300)
    cat(sprintf("  Saved: %s\n", name))
    invisible(p)
}

# ── Fig 1: Top 30 characters by dialogue lines ────────────────────────────────
top30 <- df_gpt %>%
    count(character, name = "lines") %>%
    slice_max(lines, n = 30) %>%
    mutate(character = fct_reorder(character, lines))

p1 <- ggplot(top30, aes(x = lines, y = character)) +
    geom_col(fill = col_primary, width = 0.72) +
    geom_text(aes(label = lines), hjust = -0.15, size = 2.8, color = "grey25") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
    # ggtitle("Top 30 Characters by Dialogue Lines") +
    labs(x = "Number of Lines", y = NULL) +
    theme_journal()

save_fig(p1, "fig1_top30_characters", 6.5, 8)

# ── Fig 2: Mean factor scores by episode ─────────────────────────────────────
# Derive episode-level means from panel (weighted by n_lines per character)
agg_ep <- agg_panel %>%
    group_by(season, episode, season_f, ep_order) %>%
    summarise(
        across(ends_with("_mean"), ~ weighted.mean(., w = n_lines, na.rm = TRUE)),
        .groups = "drop"
    )

# Lines segmented by season (4 sequential blue shades; same hue = one palette)
ep_long <- agg_ep %>%
    select(season_f, ep_order, ends_with("_mean")) %>%
    pivot_longer(ends_with("_mean"), names_to = "factor_key", values_to = "score") %>%
    mutate(
        factor_key = str_remove(factor_key, "_mean"),
        factor_lbl = factor(factor_labels[factor_key], levels = factor_labels)
    )

p2 <- ggplot(ep_long, aes(x = ep_order, y = score,
                           color = season_f, group = season_f)) +
    geom_line(linewidth = 0.65, alpha = 0.9) +
    geom_point(size = 1.1, alpha = 0.75) +
    facet_wrap(~ factor_lbl, ncol = 2, scales = "free_y") +
    scale_color_manual(values = season_pal, name = NULL) +
    # ggtitle("Mean Factor Scores by Episode") +
    labs(x = "Episode (sequential order)", y = "Mean Factor Score") +
    theme_journal() +
    guides(color = guide_legend(nrow = 1))

save_fig(p2, "fig2_factor_scores_by_episode", 9, 6)

# ── Fig 3: Factor score distributions by character (top 20, ≥30 lines) ───────
top_chars <- agg_panel %>%
    group_by(character) %>%
    summarise(n_lines = sum(n_lines), .groups = "drop") %>%
    filter(n_lines >= 30) %>%
    slice_max(n_lines, n = 20) %>%
    pull(character)

char_long <- df_scored %>%
    filter(character %in% top_chars) %>%
    select(character, all_of(all_score_names)) %>%
    pivot_longer(all_of(all_score_names),
                 names_to = "factor_key", values_to = "score") %>%
    mutate(
        factor_lbl = factor(factor_labels[factor_key], levels = factor_labels),
        character  = fct_reorder(character, score, .fun = median, .desc = TRUE)
    )

p3 <- ggplot(char_long, aes(x = score, y = character)) +
    geom_boxplot(
        fill         = col_primary,
        color        = "grey20",
        outlier.size  = 0.4,
        outlier.alpha = 0.3,
        width        = 0.65,
        alpha        = 0.75,
        linewidth    = 0.3
    ) +
    facet_wrap(~ factor_lbl, ncol = 2, scales = "free_x") +
    # ggtitle("Factor Score Distributions by Character") +
    labs(x = "Factor Score", y = NULL) +
    theme_journal()

save_fig(p3, "fig3_factor_scores_by_character", 9, 7)

# ── Fig 4: Transcript line length distribution ────────────────────────────────
df_gpt    <- df_gpt %>% mutate(line_length = nchar(sentence))
med_len   <- median(df_gpt$line_length, na.rm = TRUE)
p99_len   <- quantile(df_gpt$line_length, 0.99, na.rm = TRUE)

p4 <- ggplot(df_gpt, aes(x = line_length)) +
    geom_histogram(fill = col_primary, color = "white",
                   bins = 55, linewidth = 0.15) +
    geom_vline(xintercept = med_len,
               color = col_secondary, linetype = "dashed", linewidth = 0.8) +
    annotate("text",
             x = med_len + 15, y = Inf, vjust = 1.7, hjust = 0,
             size = 3, color = col_secondary,
             label = sprintf("Median = %d chars", as.integer(med_len))) +
    scale_x_continuous(limits  = c(0, p99_len),
                       expand  = expansion(mult = c(0, 0.02))) +
    scale_y_continuous(expand  = expansion(mult = c(0, 0.05))) +
    # ggtitle("Distribution of Transcript Line Lengths") +
    labs(x = "Characters per Line", y = "Count") +
    theme_journal()

save_fig(p4, "fig4_line_length_distribution", 7, 4.5)

# ── Fig 5 (additional): Correlation heatmap of all 11 indicators ──────────────
cor_mat  <- cor(df_clean[c(md_cols, inout_cols)], use = "complete.obs")
all_labs <- short_labels[c(md_cols, inout_cols)]
dimnames(cor_mat) <- list(all_labs, all_labs)

cor_long <- as.data.frame(as.table(cor_mat)) %>%
    rename(var1 = Var1, var2 = Var2, r = Freq) %>%
    mutate(
        var1  = factor(var1, levels = rev(all_labs)),
        var2  = factor(var2, levels = all_labs),
        label = ifelse(as.character(var1) == as.character(var2),
                       "", sprintf("%.2f", r))
    ) %>%
    filter(as.integer(var1) >= as.integer(var2))

p5 <- ggplot(cor_long, aes(x = var2, y = var1, fill = r)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = label), size = 2.5, color = "grey15") +
    scale_fill_gradient2(
        low      = col_secondary,
        mid      = "white",
        high     = col_primary,
        midpoint = 0,
        limits   = c(-1, 1),
        name     = "r"
    ) +
    # ggtitle("Correlation Matrix: MD and In/Out Indicators") +
    labs(x = NULL, y = NULL) +
    theme_journal() +
    theme(
        axis.text.x     = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y     = element_text(size = 8),
        legend.position = "right"
    )

save_fig(p5, "fig5_correlation_heatmap", 8, 6.5)

# ── Fig 6 (additional): MD factor loadings heatmap ────────────────────────────
loadings_df <- as.data.frame(unclass(fa_md$loadings)) %>%
    rownames_to_column("indicator") %>%
    pivot_longer(-indicator, names_to = "factor_key", values_to = "loading") %>%
    mutate(
        ind_lbl    = factor(short_labels[indicator],
                            levels = rev(short_labels[md_cols])),
        factor_lbl = factor(factor_labels[factor_key],
                            levels = factor_labels[md_score_names]),
        label      = sprintf("%.2f", loading)
    )

p6 <- ggplot(loadings_df, aes(x = factor_lbl, y = ind_lbl, fill = loading)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = label), size = 3, color = "grey10") +
    scale_fill_gradient2(
        low      = col_secondary,
        mid      = "white",
        high     = col_primary,
        midpoint = 0,
        limits   = c(-1, 1),
        name     = "Loading"
    ) +
    # ggtitle("MD Factor Loadings (oblimin rotation, ML)") +
    labs(x = "Factor", y = NULL) +
    theme_journal() +
    theme(legend.position = "right")

save_fig(p6, "fig6_md_factor_loadings", 5.5, 5)

# ── Fig 7 (additional): Scree plot with parallel analysis ─────────────────────
scree_df <- tibble(
    component = seq_along(pa_md$fa.values),
    Observed  = pa_md$fa.values,
    Simulated = pa_md$fa.sim
) %>%
    pivot_longer(-component, names_to = "type", values_to = "eigenvalue")

p7 <- ggplot(scree_df, aes(x = component, y = eigenvalue,
                            color = type, linetype = type, shape = type)) +
    geom_hline(yintercept = 1, linetype = "dotted",
               color = "grey55", linewidth = 0.5) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 2.5) +
    scale_color_manual(
        values = c(Observed = col_primary, Simulated = col_secondary),
        name   = NULL
    ) +
    scale_linetype_manual(
        values = c(Observed = "solid", Simulated = "dashed"),
        name   = NULL
    ) +
    scale_shape_manual(
        values = c(Observed = 19, Simulated = 17),
        name   = NULL
    ) +
    scale_x_continuous(breaks = seq_along(pa_md$fa.values)) +
    # ggtitle("Scree Plot with Parallel Analysis (MD Indicators)") +
    labs(x = "Factor Number", y = "Eigenvalue") +
    theme_journal() +
    guides(
        color    = guide_legend(nrow = 1),
        linetype = guide_legend(nrow = 1),
        shape    = guide_legend(nrow = 1)
    )

save_fig(p7, "fig7_scree_plot_parallel_analysis", 6, 4)

cat("\nAll 7 figures saved to ./figures/\n")
