### 05.23.2026
### This is script is for factor analysis 

suppressPackageStartupMessages({
    library(haven)
    library(ggplot2)
    library(tidyverse)
    library(psych)
})


## Load datasets
df_gpt <- read_csv("./data/aot_gpt_mini_labeled.csv", show_col_types = F)
df_qwen <- read_csv("./data/aot_qwen_labeled.csv", show_col_types = F)

